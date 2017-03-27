
import java.io.*;
import java.util.*;

public class Data{
	public String[] scanIds;
	public Map<String, Scan> idToScan;
  private final String dataDir, solutionPath;
  
  public Data(String dataDir, String solutionPath){
    this.dataDir = dataDir; this.solutionPath = solutionPath;
  }
  
  private void log(String s){ System.out.println(s); }
  
  public void loadMetaData() {
    log("Loading scan list from " + dataDir + " ...");
    // gather scan ids
    idToScan = new HashMap<>();
    for (File f: new File(dataDir).listFiles()) {
      if (f.isDirectory() && new File(f, "auxiliary").exists()) {
        String id = f.getName();
        Scan scan = new Scan(id);
        idToScan.put(id, scan);
      }
    }
    scanIds = idToScan.keySet().toArray(new String[0]);
		Arrays.sort(scanIds);
		
		int scanCnt = scanIds.length;
    log(scanCnt + " total scans...");
		
		String line = null;
		int lineNo = 0, iScan = 0;
		
    	// load scan and slice meta data
		for (Scan scan: idToScan.values()) {
			File scanDir = new File(dataDir, scan.id);
      iScan++;
			System.out.print(iScan + " ");
			// load structure names and ids
	    	try {
				File f = new File(scanDir, "structures.dat");
				if (f.exists()) {
					LineNumberReader lnr = new LineNumberReader(new FileReader(f));
					while (true) {
						line = lnr.readLine();
						lineNo++;
						if (line == null) break;
						line = line.trim();
						if (line.isEmpty()) continue;
						// body|Esophagus|lung,radiomics_gtv 
				    	String[] parts = line.split("\\|");
				    	for (int i = 0; i < parts.length; i++) {
				    		String contourName = parts[i];
				    		if (G.CONTOUR_TRUTH_ALIASES.contains(contourName)) {
				    			contourName = G.CONTOUR_TRUTH;
				    		}
				    		// structure ids are 1-based
				    		scan.structureIdToName.put(i+1, contourName);
				    	}
					}
					lnr.close();
				}
			} 
			catch (Exception e) {
				log("Error reading structures.dat file for scan " + scan.id);
				log("Line #" + lineNo + ": " + line);
				System.err.println(e);
				System.exit(0);
			}
			
			File dir = new File(scanDir, "auxiliary");
			for (File f: dir.listFiles()) {
				// name is like 103.dat
				String[] nameParts = f.getName().split("\\.");
				int sliceOrdinal = Integer.parseInt(nameParts[0]);
				if (scan.slices.size() < sliceOrdinal) {
					int missing = sliceOrdinal - scan.slices.size();
					for (int i = 0; i < missing; i++) {
						scan.slices.add(new Slice());
					}
				}
				Slice slice = scan.slices.get(sliceOrdinal - 1);
		    line = null;
				lineNo = 0;
				try {
					LineNumberReader lnr = new LineNumberReader(new FileReader(f));
					while (true) {
						line = lnr.readLine();
						lineNo++;
						if (line == null) break;
						line = line.trim();
						/*
						(0020.0032),-250,-250,-47.5
						(0020.0037),1,0,0,0,1,0
						(0018.0050),2.5
						(0028.0010),512
						(0028.0011),512
						(0028.0030),9.76562e-1,9.76562e-1
						(0028.1052),-1024
						(0028.1053),1
						(0028.1054),HU
						*/
						String[] parts = line.split(",");
						String tag = parts[0];
						if (tag.equals("(0020.0032)")) {
							slice.x0 = Double.parseDouble(parts[1]);
							slice.y0 = Double.parseDouble(parts[2]);
							slice.z  = Double.parseDouble(parts[3]);
						}
						else if (tag.equals("(0028.0010)")) {
							slice.w = Integer.parseInt(parts[1]);
						}
						else if (tag.equals("(0028.0011)")) { 
							slice.h = Integer.parseInt(parts[1]);
						}
						else if (tag.equals("(0028.0030)")) {
							slice.dx = Double.parseDouble(parts[1]);
							slice.dy = Double.parseDouble(parts[2]);
						}
						else if (tag.equals("(0018.0050)")) {
							slice.dz = Double.parseDouble(parts[1]);
						}
					}
					lnr.close();
				} 
				catch (Exception e) {
					log("Error reading auxiliary file for : " + scan.id);
					log("Line #" + lineNo + ": " + line);
					e.printStackTrace();
					System.exit(0);
				}
			} // for .aux files
		
			// load contours
			dir = new File(scanDir, "contours");
			loadContours(scan, dir, true); // only TRUTH now
		} // for scans
    log("\n" + scanCnt + " scans read.");
	}
  
  public void loadSolution() {
    File f = new File(solutionPath);
		if (f.exists()) {
			log("Loading solution file from " + solutionPath);
			String line = null;
			int lineNo = 0;
			try {
				LineNumberReader lnr = new LineNumberReader(new FileReader(f));
				while (true) {
					line = lnr.readLine();
					lineNo++;
					if (line == null) break;
					line = line.trim();
					if (line.isEmpty()) continue;
					// ANON_LUNG_TC001,100,x1,y1,x2,y2,...
					String[] parts = line.split(",");
					String id = parts[0];
					Scan scan = idToScan.get(id);
					if (scan == null) {
						log("Unknown scan id found in solution file at line " + lineNo + ": " + id);
						System.exit(0);
					}
					
					int sliceOrdinal = Integer.parseInt(parts[1]);
					if (scan.slices.size() < sliceOrdinal) {
						log("Unknown slice id found in solution file at line " + lineNo + ": " + id + ", " + sliceOrdinal);
						System.exit(0);
					}
					Slice slice = scan.slices.get(sliceOrdinal - 1);
					
					String contourName = G.CONTOUR_SOLUTION;
					List<Polygon> polygons = slice.nameToPolygons.get(contourName);
					if (polygons == null) {
						polygons = new Vector<>();
						slice.nameToPolygons.put(contourName, polygons);
					}
					
					int skip = parts[0].length() + parts[1].length() + 2;
					line = line.substring(skip);
					P2[] points = G.coordStringToPoints(line, 2); // nCoords is 2: x,y
					// convert to pixels
					int n = points.length;
					for (int i = 0; i < n; i++) {
						points[i] = G.mmToPixel(points[i], slice);
					}
			    	Polygon p = new Polygon(points); 
					polygons.add(p);
				}
				lnr.close();
			} 
			catch (Exception e) {
				log("Error reading solution file");
				log("Line #" + lineNo + ": " + line);
				e.printStackTrace();
				System.exit(0);
			}
		}
		else {
			log("Can't find solution file " + f.getAbsolutePath());
		}
	}
  
  private void loadContours(Scan scan, File dir, boolean truth) {
    if (dir == null || !dir.exists() || !dir.isDirectory()) return;
    for (File f: dir.listFiles()) {
      // name is like 100.1.dat
      String[] nameParts = f.getName().split("\\.");
			int sliceOrdinal = Integer.parseInt(nameParts[0]);
			if (scan.slices.size() < sliceOrdinal) {
				log("Slice " + sliceOrdinal + " doesn't exist for scan: " + scan.id);
				System.exit(0);
			}
			Slice slice = scan.slices.get(sliceOrdinal - 1);
			
			int contourId = Integer.parseInt(nameParts[1]);
			String contourName = scan.structureIdToName.get(contourId);
			if (contourName == null) {
				log("Contour Id " + contourId + " not known for scan: " + scan.id);
				System.exit(0);
			}
			
			if (truth && !contourName.equals(G.CONTOUR_TRUTH)) continue;
			if (!truth && contourName.equals(G.CONTOUR_TRUTH)) continue;
			
			scan.structureNames.add(contourName);
			
			List<Polygon> polygons = slice.nameToPolygons.get(contourName);
			if (polygons == null) {
				polygons = new Vector<>();
				slice.nameToPolygons.put(contourName, polygons);
			}
			
	    String line = null;
			int lineNo = 0;
			try {
				LineNumberReader lnr = new LineNumberReader(new FileReader(f));
				while (true) {
					line = lnr.readLine();
					lineNo++;
					if (line == null) break;
					line = line.trim();
					P2[] points = G.coordStringToPoints(line, 3); // nCoords is 3: x,y,z
					int n = points.length;
					// convert to pixels
					for (int i = 0; i < n; i++) {
						points[i] = G.mmToPixel(points[i], slice);
					}
			    	Polygon p = new Polygon(points);
					polygons.add(p);
				}
				lnr.close();
			} 
			catch (Exception e) {
				log("Error reading contour file " + f.getName() + " for scan: " + scan.id);
				log("Line #" + lineNo + ": " + line);
				System.exit(0);
			}
		}
  }
}

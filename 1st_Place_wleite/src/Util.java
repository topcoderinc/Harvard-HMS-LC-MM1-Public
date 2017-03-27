import java.awt.Point;
import java.awt.Polygon;
import java.awt.geom.Area;
import java.awt.geom.Line2D;
import java.awt.geom.PathIterator;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Util {
	private static String[] tumorStructures = "radiomics_gtv|radiomics_gtv|Radiomics_gtv|radiomics_gtv2|radiomics_gtv_nw|radiomics_gtvr".split("\\|");
	private static String[] lungsStructures = "lungs|Both Lungs|Combo Lung|COMBO LUNGS|lung|Lung|LUNG RT and LT|lungs|Lungs|LUNGS|Lungs_Total|Lungs_TOTAL|Lung_Total|LUNG TOTAL|total lung|Total lung|Total Lung|TOTAL LUNG|TOTAL LUNG.|total lungs|Total Lungs|TOT LUNG|whole lung|RT and LT Lungs|RT and LT LUNGS"
			.split("\\|");

	public static byte[][] evalImage(SliceImage imagePrev, SliceImage image, SliceImage imageNext, Slice slice, RandomForestPredictor predictor, int usedContrast, double slicePct) {
		try {
			int w = image.width;
			int h = image.height;
			byte[][] ret = new byte[h][w];
			TumorFeatureExtractor ext = new TumorFeatureExtractor(imagePrev, image, imageNext, slice, usedContrast, slicePct);
			for (int y = 0; y < h; y++) {
				for (int x = 0; x < w; x++) {
					if (image.gray[y * w + x] == 0) continue;
					float[] features = ext.getFeatures(x, y);
					ret[y][x] = (byte) Math.round(255 * predictor.predict(features));
				}
			}
			return ret;
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-11);
		}
		return null;
	}

	public static Slice readSlice(File file) {
		Slice slice = new Slice();
		try {
			BufferedReader in = new BufferedReader(new FileReader(file));

			String line = null;
			while ((line = in.readLine()) != null) {
				String[] parts = line.trim().split(",");
				String tag = parts[0];
				if (tag.equals("(0020.0032)")) {
					slice.x0 = Double.parseDouble(parts[1]);
					slice.y0 = Double.parseDouble(parts[2]);
					slice.z = Double.parseDouble(parts[3]);
				} else if (tag.equals("(0028.0010)")) {
					slice.w = Integer.parseInt(parts[1]);
				} else if (tag.equals("(0028.0011)")) {
					slice.h = Integer.parseInt(parts[1]);
				} else if (tag.equals("(0028.0030)")) {
					slice.dx = Double.parseDouble(parts[1]);
					slice.dy = Double.parseDouble(parts[2]);
				} else if (tag.equals("(0018.0050)")) {
					slice.dz = Double.parseDouble(parts[1]);
				}
			}
			in.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-6);
		}
		return slice;
	}

	public static int[] findLungRange(File[] auxFiles, File folder, String patient, RandomForestPredictor lungPredictor, int usedContrast) {
		double[] v = new double[auxFiles.length + 1];
		for (File auxFile : auxFiles) {
			int p = auxFile.getName().indexOf('.');
			int sliceId = Integer.parseInt(auxFile.getName().substring(0, p));
			File imageFile = new File(folder, patient + "/pngs/" + sliceId + ".png");
			Slice slice = Util.readSlice(auxFile);
			v[sliceId] = evalContainLung(imageFile, slice, lungPredictor, usedContrast);
			//System.err.println(sliceId + "\t" + v[sliceId]);
		}
		int min = v.length - 1;
		int max = 1;
		for (int i = 0; i < v.length; i++) {
			if (v[i] > 0.8) {
				min = Math.min(i, min);
				max = Math.max(i, max);
			}
		}
		for (int i = min - 1; i >= 0; i--) {
			if (v[i] > 0.8) min = i;
			else break;
		}
		if (max - min < 10) return new int[] { 1, v.length - 1 };
		for (int i = max + 1; i < v.length; i++) {
			if (v[i] > 0.5) max = i;
			else break;
		}
		for (int i = 0; i < 4; i++) {
			if (min > 1) min--;
			if (max < v.length - 1) max++;
		}
		return new int[] { min, max };
	}

	private static double evalContainLung(File imageFile, Slice slice, RandomForestPredictor lungPredictor, int usedContrast) {
		SliceImage image = new SliceImage(imageFile, true);
		float[] features = new LungFeatureExtractor().getFeatures(image, slice, usedContrast);
		return lungPredictor.predict(features);
	}

	public static void updateImagesCenter(File[] auxFiles, File folder, String patient, int[] lungRange) {
		List<File> files = new ArrayList<File>();
		List<Integer> xc0 = new ArrayList<Integer>();
		List<Integer> xc1 = new ArrayList<Integer>();
		List<Integer> yc = new ArrayList<Integer>();
		for (File auxFile : auxFiles) {
			int p = auxFile.getName().indexOf('.');
			int sliceId = Integer.parseInt(auxFile.getName().substring(0, p));
			if (sliceId < lungRange[0] || sliceId > lungRange[1]) continue;
			File imageFile = new File(folder, patient + "/pngs/" + sliceId + ".png");
			SliceImage image = new SliceImage(imageFile, true);
			int[] xc = image.findCenterMirror();
			files.add(imageFile);
			yc.add(image.yCenter);
			xc0.add(xc[0]);
			xc1.add(xc[1]);
		}
		for (int i = 0; i < files.size(); i++) {
			int x0 = 0;
			int x1 = 0;
			int div = 0;
			for (int j = i - 2; j <= i + 2; j++) {
				if (j < 0 || j >= xc0.size()) continue;
				x0 += xc0.get(j);
				x1 += xc1.get(j);
				div++;
			}
			x0 /= div;
			x1 /= div;
			SliceImage.fixCenter(files.get(i), x0, x1, yc.get(i));
		}
	}

	/*
	public static Point findBodyCenter(File[] auxFiles, File folder, String patient) {
		List<Integer> xc = new ArrayList<Integer>();
		List<Integer> yc = new ArrayList<Integer>();
		for (File auxFile : auxFiles) {
			int p = auxFile.getName().indexOf('.');
			int sliceId = Integer.parseInt(auxFile.getName().substring(0, p));
			File imageFile = new File(folder, patient + "/pngs/" + sliceId + ".png");
			SliceImage image = new SliceImage(imageFile, true, null);
			xc.add(image.bodyCenter.x);
			yc.add(image.bodyCenter.y);
		}
		int mid = xc.size() / 2;
		Collections.sort(xc);
		Collections.sort(yc);
		return new Point(xc.get(mid), yc.get(mid));
	}
	*/
	public static File[] getContourFiles(File folder, String patient) {
		File d = new File(folder, patient + "/contours");
		if (!d.exists()) return new File[0];
		return d.listFiles();
	}

	public static File[] getAuxFiles(File folder, String patient) {
		File[] auxFiles = new File(folder, patient + "/auxiliary").listFiles();
		Arrays.sort(auxFiles, new Comparator<File>() {
			public int compare(File a, File b) {
				return Integer.compare(c(a), c(b));
			}

			private int c(File a) {
				int p = a.getName().indexOf('.');
				return Integer.parseInt(a.getName().substring(0, p));
			}
		});
		return auxFiles;
	}

	public static int findTumorStructuresIndex(File structuresFile) {
		return findStructuresIndex(structuresFile, tumorStructures);
	}

	public static int findLungsStructuresIndex(File structuresFile) {
		return findStructuresIndex(structuresFile, lungsStructures);
	}

	private static int findStructuresIndex(File structuresFile, String[] structures) {
		if (!structuresFile.exists()) return -1;
		try {
			BufferedReader in = new BufferedReader(new FileReader(structuresFile));
			String line = in.readLine();
			in.close();
			String[] s = line.split("\\|");
			for (int i = 0; i < s.length; i++) {
				for (int j = 0; j < structures.length; j++) {
					if (structures[j].equalsIgnoreCase(s[i])) return i + 1;
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-2);
		}
		return 0;
	}

	public static List<Region> extractRegions(File file, Slice slice, int sliceId) {
		List<Region> regions = new ArrayList<Region>();
		try {
			BufferedReader in = new BufferedReader(new FileReader(file));
			String line = null;
			while ((line = in.readLine()) != null) {
				String[] s = line.split(",");
				int n = s.length / 3;
				int[] xp = new int[n];
				int[] yp = new int[n];
				int j = 0;
				for (int i = 0; i < n; i++, j++) {
					Point p = mmToPixel(Double.parseDouble(s[j++]), Double.parseDouble(s[j++]), slice);
					xp[i] = p.x;
					yp[i] = p.y;
				}
				Region region = new Region(slice, sliceId);
				region.in = new Polygon(xp, yp, n);
				if (region.getAreaVal() < 5) continue;
				regions.add(region);
			}

			in.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-3);
		}
		return regions;
	}

	public static Point pixelToMm(int x, int y, Slice slice) {
		x = (int) Math.round((x * slice.dx) + slice.x0);
		y = (int) Math.round((y * slice.dy) + slice.y0);
		return new Point(x, y);
	}

	public static int pixelToX(int x, Slice slice) {
		return (int) Math.round((x * slice.dx) + slice.x0);
	}

	public static int pixelToY(int y, Slice slice) {
		return (int) Math.round((y * slice.dy) + slice.y0);
	}

	public static Point mmToPixel(double x, double y, Slice slice) {
		int px = (int) Math.round((x - slice.x0) / slice.dx);
		int py = (int) Math.round((y - slice.y0) / slice.dy);
		return new Point(px, py);
	}

	public static int xToPixel(double x, Slice slice) {
		return (int) Math.round((x - slice.x0) / slice.dx);
	}

	public static int yToPixel(double y, Slice slice) {
		return (int) Math.round((y - slice.y0) / slice.dy);
	}

	public static List<String> split(List<String> in, double cut, boolean first) {
		List<String> out = new ArrayList<String>(in);
		Random rnd = new Random(20140819);
		for (int i = out.size(); i > 1; i--) {
			Collections.swap(out, i - 1, rnd.nextInt(i));
		}
		int pos = (int) Math.round(in.size() * cut);
		if (first) out.subList(pos, out.size()).clear();
		else out.subList(0, pos).clear();
		return out;
	}

	public static List<String> readContent(File folder) {
		List<String> ret = new ArrayList<String>();
		try {
			File[] files = folder.listFiles();
			for (File file : files) {
				ret.add(file.getName());
			}
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
		return ret;
	}

	public static Map<String, Integer> readClinical(File folder) {
		Map<String, Integer> ret = new HashMap<String, Integer>();
		try {
			File[] files = folder.listFiles();
			for (File file : files) {
				BufferedReader in = new BufferedReader(new FileReader(file));
				String line = null;
				while ((line = in.readLine()) != null) {
					String[] s = line.split(",");
					if (s.length >= 2) ret.put(s[0], Integer.parseInt(s[1]));
				}
				in.close();
			}
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
		return ret;
	}

	public static double areaVal(Area shape) {
		PathIterator i = shape.getPathIterator(null);
		double a = 0.0;
		double[] coords = new double[6];
		double startX = Double.NaN, startY = Double.NaN;
		Line2D segment = new Line2D.Double(Double.NaN, Double.NaN, Double.NaN, Double.NaN);
		while (!i.isDone()) {
			int segType = i.currentSegment(coords);
			double x = coords[0], y = coords[1];
			switch (segType) {
			case PathIterator.SEG_CLOSE:
				segment.setLine(segment.getX2(), segment.getY2(), startX, startY);
				a += area(segment);
				startX = startY = Double.NaN;
				segment.setLine(Double.NaN, Double.NaN, Double.NaN, Double.NaN);
				break;
			case PathIterator.SEG_LINETO:
				segment.setLine(segment.getX2(), segment.getY2(), x, y);
				a += area(segment);
				break;
			case PathIterator.SEG_MOVETO:
				startX = x;
				startY = y;
				segment.setLine(Double.NaN, Double.NaN, x, y);
				break;
			}
			i.next();
		}
		if (Double.isNaN(a)) {
			throw new IllegalArgumentException("PathIterator contains an open path");
		} else {
			return 0.5 * Math.abs(a);
		}
	}

	private static double area(Line2D seg) {
		return seg.getX1() * seg.getY2() - seg.getX2() * seg.getY1();
	}
}

class Segment {
	final int x0, y0, x1, y1, dx, dy;
	final int xMin, yMin, xMax, yMax;

	public Segment(int x0, int y0, int x1, int y1) {
		this.x0 = x0;
		this.y0 = y0;
		this.x1 = x1;
		this.y1 = y1;
		dx = x1 - x0;
		dy = y1 - y0;
		if (x0 < x1) {
			xMin = x0;
			xMax = x1;
		} else {
			xMin = x1;
			xMax = x0;
		}
		if (y0 < y1) {
			yMin = y0;
			yMax = y1;
		} else {
			yMin = y1;
			yMax = y0;
		}
	}

	public int intersects(Segment o) {
		if (xMin > o.xMax) return 0;
		if (xMax < o.xMin) return 0;
		if (yMin > o.yMax) return 0;
		if (yMax < o.yMin) return 0;

		boolean b0 = (o.x0 == x0 && o.y0 == y0) || (o.x1 == x0 && o.y1 == y0);
		boolean b1 = (o.x0 == x1 && o.y0 == y1) || (o.x1 == x1 && o.y1 == y1);
		if (b0 && b1) return 2;
		int den = o.dy * dx - o.dx * dy;
		int num1 = o.dx * (y0 - o.y0) - o.dy * (x0 - o.x0);
		int num2 = dx * (y0 - o.y0) - dy * (x0 - o.x0);
		if (den == 0 && (num1 == 0 || num2 == 0)) {
			return 2;
		}
		if (den == 0) return 0;
		if (den < 0) {
			den = -den;
			num1 = -num1;
			num2 = -num2;
		}
		if (num1 >= 0 && num2 >= 0 && num1 <= den && num2 <= den) {
			if (b0 || b1) return 0;
			return 2;
		}
		return 0;
	}
}

class Slice {
	public int w, h;
	public double x0, y0, z;
	public double dx, dy, dz;
}

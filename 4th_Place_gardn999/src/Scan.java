
import java.awt.geom.Area;
import java.util.*;

public class Scan {
  public String id;
  public List<Slice> slices = new ArrayList<>();
  public Set<String> structureNames = new HashSet<>();
  public Map<Integer, String> structureIdToName = new HashMap<>();
  
  public Scan(String id) {
    this.id = id;
  }
  
  public double scoreTotal(){
    Metric[] result = sliceScores();
    if (result == null) return 0;
    double tp = 0, fp = 0, fn = 0;
    for (Metric m : result){
      if (m != null) { tp += m.tp; fp += m.fp; fn += m.fn; }
    }
    double score = 0;
    if (tp > 0) {
      double t = tp + fn;
      double fn2 = fn * t / tp;
      double e = fn2 + fp;
      double a = Math.pow(36 * Math.PI * t * t, (double)1 / 3);
      double exp1 = e / t;
      double exp2 = e / (10 * a);
      score = Math.exp(-(exp1 + exp2)/2);
    }
    return score;
  }
  
  public Metric[] sliceScores(){
		Metric[] ret = new Metric[slices.size()];
		for (int i = 0; i < slices.size(); i++) {
			Slice slice = slices.get(i);
			double areaTruth = 0;
			List<Polygon> truthPolygons = slice.nameToPolygons.get(G.CONTOUR_TRUTH);
			if (truthPolygons != null) {
				for (Polygon p: truthPolygons) areaTruth += p.area;
			}
			double areaSolution = 0;
			List<Polygon> solutionPolygons = slice.nameToPolygons.get(G.CONTOUR_SOLUTION);
			if (solutionPolygons != null) {
				for (Polygon p: solutionPolygons) areaSolution += p.area;
			}
			Metric m = new Metric();
			if (areaTruth == 0) { 
				if (areaSolution == 0) { // neither exist
					ret[i] = null;
					continue;
				}
				else { // no truth, false sol
					m.fp = areaSolution;
				}
			}
			else {
				if (areaSolution == 0) { // truth, no sol
					m.fn = areaTruth;
				}
				else { // both exist, calc tp,fp,fn
					Area shapeT = new Area();
					for (Polygon p: truthPolygons) shapeT.add(p.shape);
					Area shapeS = new Area();
					for (Polygon p: solutionPolygons) shapeS.add(p.shape);
					shapeT.intersect(shapeS);
					double overlap = Math.abs(G.area(shapeT));
					m.tp = overlap;
					m.fp = areaSolution - overlap;
					m.fn = areaTruth - overlap;
				}
			}
			// multiply areas with voxel volume
			double v = slice.dx * slice.dy * slice.dz;
			m.tp *= v; m.fp *= v; m.fn *= v;
			ret[i] = m;
		}
		return ret;
  }
}

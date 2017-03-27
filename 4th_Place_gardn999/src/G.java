
import java.awt.geom.*;
import java.text.*;
import java.util.*;

public class G{
	public static final String CONTOUR_TRUTH = "radiomics_gtv";
	public static final String CONTOUR_SOLUTION = "solution";
	public static final Set<String> CONTOUR_TRUTH_ALIASES;
	static {
		String[] aliases = {
			"radiomics_gtv", "Radiomics_gtv", "radiomics_gtv2", "radiomics_gtv_nw", "radiomics_gtvr"
		};
		CONTOUR_TRUTH_ALIASES = new HashSet<>();
    CONTOUR_TRUTH_ALIASES.addAll(Arrays.asList(aliases));
	}
	
  
	private static final DecimalFormat df; 
	private static final DecimalFormat df6; 
	static {
		df = new DecimalFormat("0.###");
		df6 = new DecimalFormat("0.######");
		DecimalFormatSymbols dfs = new DecimalFormatSymbols();
		dfs.setDecimalSeparator('.');
		df.setDecimalFormatSymbols(dfs);
		df6.setDecimalFormatSymbols(dfs);		
	}

	/**
	 * Pretty print a double
	 */
	public static String f(double d) {
		return df.format(d);
	}
	public static String f6(double d) {
		return df6.format(d);
	}

	// Convert from image space (pixels) to physical space (millimeters)
	public static P2 pixelToMm(P2 p, Slice slice) {
		double x = (p.x * slice.dx) + slice.x0;
		double y = (p.y * slice.dy) + slice.y0;
		return new P2(x, y);
	}
	
	// Convert from physical space (millimeters) to image space (pixels)
	public static P2 mmToPixel(P2 p, Slice slice) {
		double x = (p.x - slice.x0) / slice.dx;
		double y = (p.y - slice.y0) / slice.dy;
		return new P2(x, y);
	}
	
	public static P2[] coordStringToPoints(String coordString, int nCoords) {
		// x1,y1,z1,x2,y2,z2,... or x1,y1,x2,y2,... depending on nCoords
		String[] parts = coordString.split(",");
		int n = parts.length / nCoords; 
		P2[] points = new P2[n];
		for (int i = 0; i < n; i++) {
			double x = Double.parseDouble(parts[nCoords*i]);
			double y = Double.parseDouble(parts[nCoords*i+1]);
			points[i] = new P2(x, y);
		}
		return points;		
	}
  
	// based on http://stackoverflow.com/questions/2263272/how-to-calculate-the-area-of-a-java-awt-geom-area
	public static double area(Area shape) {
		PathIterator i = shape.getPathIterator(null);
		double a = 0.0;
    double[] coords = new double[6];
    double startX = Double.NaN, startY = Double.NaN;
    Line2D segment = new Line2D.Double(Double.NaN, Double.NaN, Double.NaN, Double.NaN);
    while (! i.isDone()) {
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

  public static double area(Line2D seg) {
    return seg.getX1() * seg.getY2() - seg.getX2() * seg.getY1();
  }
}

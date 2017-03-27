import java.awt.Point;
import java.awt.Polygon;
import java.util.ArrayList;
import java.util.List;

public class SimplifyPolygon {
	public static Polygon simplify(Polygon poly, int tolerance) {
		if (poly == null || poly.npoints < 3) return poly;
		List<Point> pts = new ArrayList<Point>();
		for (int i = 0; i < poly.npoints; i++) {
			pts.add(new Point(poly.xpoints[i], poly.ypoints[i]));
		}
		while (pts.size() > 3) {
			int min = tolerance;
			Point p1 = pts.get(pts.size() - 2);
			Point p2 = pts.get(pts.size() - 1);
			int rem = -1;
			for (int i = 0; i < pts.size(); i++) {
				Point p3 = pts.get(i);
				int curr = triangleArea(p1, p2, p3);
				if (curr < min) {
					min = curr;
					rem = i;
				}
				p1 = p2;
				p2 = p3;
			}
			if (rem == -1) break;
			pts.remove(rem == 0 ? pts.size() - 1 : rem - 1);
		}
		if (pts.size() >= poly.npoints) return poly;
		int[] xp = new int[pts.size()];
		int[] yp = new int[pts.size()];
		for (int i = 0; i < xp.length; i++) {
			Point p = pts.get(i);
			xp[i] = p.x;
			yp[i] = p.y;
		}
		return new Polygon(xp, yp, xp.length);
	}

	private static int triangleArea(Point p1, Point p2, Point p3) {
		return Math.abs(p1.x * p2.y + p1.y * p3.x + p2.x * p3.y - p3.x * p2.y - p1.y * p2.x - p1.x * p3.y);
	}
}
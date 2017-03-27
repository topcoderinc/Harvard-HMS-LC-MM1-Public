import java.awt.Polygon;
import java.awt.Rectangle;
import java.awt.geom.AffineTransform;
import java.awt.geom.Area;
import java.awt.geom.Point2D;
import java.io.Serializable;

public class Region implements Serializable {
	private static final long serialVersionUID = -8096014367711568204L;
	Polygon in;
	private transient Area area;
	private double areaVal = -1;
	private double[] rectLen;
	double value;
	final int sliceId;
	final Slice slice;

	public Region(Slice slice, int sliceId) {
		this.slice = slice;
		this.sliceId = sliceId;
	}

	public void invalidate() {
		area = null;
		areaVal = -1;
		rectLen = null;
	}

	public Area getArea() {
		if (area == null) area = new Area(in);
		return area;
	}

	public double getAreaVal() {
		if (areaVal == -1) areaVal = Util.areaVal(getArea());
		return areaVal;
	}

	public double getMinDim() {
		if (rectLen == null) rectLen = rectLen(in);
		return Math.min(rectLen[0], rectLen[1]);
	}

	public double getMaxDim() {
		if (rectLen == null) rectLen = rectLen(in);
		return Math.max(rectLen[0], rectLen[1]);
	}

	public double getBoxProp() {
		if (rectLen == null) rectLen = rectLen(in);
		double a = rectLen[0] * (double) rectLen[1];
		return a == 0 ? 0 : getAreaVal() / a;
	}

	public double getXYProp() {
		if (rectLen == null) rectLen = rectLen(in);
		double a = Math.min(rectLen[0], rectLen[1]);
		double b = Math.max(rectLen[0], rectLen[1]);
		return b == 0 ? 0 : a / b;
	}

	private static double[] rectLen(Polygon polygon) {
		Rectangle rc = polygon.getBounds();
		double xc = rc.getCenterX();
		double yc = rc.getCenterY();
		Point2D[] org = new Point2D[polygon.npoints];
		Point2D[] dst = new Point2D[polygon.npoints];
		for (int i = 0; i < polygon.npoints; i++) {
			org[i] = new Point2D.Double(polygon.xpoints[i], polygon.ypoints[i]);
		}
		double minArea = 1e99;
		double[] ret = new double[2];
		for (int a = 0; a < 90; a += 3) {
			AffineTransform.getRotateInstance(Math.toRadians(a), xc, yc).transform(org, 0, dst, 0, org.length);
			Point2D r = dst[0];
			double xmin = r.getX();
			double ymin = r.getY();
			double xmax = xmin;
			double ymax = ymin;
			for (int i = 1; i < polygon.npoints; i++) {
				r = dst[i];
				xmin = Math.min(xmin, r.getX());
				xmax = Math.max(xmax, r.getX());
				ymin = Math.min(ymin, r.getY());
				ymax = Math.max(ymax, r.getY());
			}
			double dx = xmax - xmin;
			double dy = ymax - ymin;
			double currArea = dx * dy;
			if (currArea < minArea) {
				minArea = currArea;
				ret[0] = Math.min(dx, dy);
				ret[1] = Math.max(dx, dy);
			}
		}
		return ret;
	}
}

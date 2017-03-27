
import java.awt.geom.*;

public class Polygon {
  public double minx, miny, maxx, maxy;
  public double area = 0;
  public Area shape;
  public P2[] points;
  
  public Polygon(P2[] points) {
    this.points = points;
    makeBounds();
    getShape();
    getArea();
  }
  
  private void makeBounds() {
    minx = Double.MAX_VALUE;
    miny = Double.MAX_VALUE;
    maxx = -Double.MAX_VALUE;
    maxy = -Double.MAX_VALUE;
    for (P2 p: points) {
      minx = Math.min(p.x, minx);
      maxx = Math.max(p.x, maxx);
      miny = Math.min(p.y, miny);
      maxy = Math.max(p.y, maxy);
    }
  }
  
  public final Area getShape() {
    if (shape == null) {
      Path2D path = new Path2D.Double();
      path.setWindingRule(Path2D.WIND_EVEN_ODD);
      
      int n = points.length;
      path.moveTo(points[0].x, points[0].y);
      for(int i = 1; i < n; ++i) {
        path.lineTo(points[i].x, points[i].y);
      }
      path.closePath();
      shape = new Area(path);
    }
    return shape;
  }
  
  private void getArea() {
    // unsigned area calculated from the points
    double a = 0;
    int n = points.length;
    for (int i = 1; i < n; i++) {
      a += (points[i-1].x + points[i].x) * (points[i-1].y - points[i].y);
    }
    // process last segment if ring is not closed
    if (!points[0].equals(points[n-1])) {
      a += (points[n-1].x + points[0].x) * (points[n-1].y - points[0].y);
    }
    area = Math.abs(a / 2);
  }
  
  @Override
  public String toString() {
    return G.f(minx) + "," + G.f(miny) + " - " + 
         G.f(maxx) + "," + G.f(maxy);
  }
}


public class P2 {
  public double x;
  public double y;
  
  public P2(double x, double y) {
    this.x = x; this.y = y;
  }
  
  @Override
  public String toString() {
    return G.f(x) + ", " + G.f(y);
  }
  
  @Override
  public boolean equals(Object o) {
    if (!(o instanceof P2)) return false;
    P2 p = (P2)o;
    double d2 = (x - p.x) * (x - p.x) + (y - p.y) * (y - p.y);
    return d2 < 1e-6;
  }
}
	

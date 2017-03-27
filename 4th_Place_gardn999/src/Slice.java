
import java.util.*;

public class Slice {
  public int w, h; // image size
  public double x0, y0, z; // real space (mm)
  public double dx, dy, dz; // pixel size (mm / pixel)
  public Map<String, List<Polygon>> nameToPolygons = new HashMap<>();
}

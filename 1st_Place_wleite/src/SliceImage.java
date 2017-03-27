import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import javax.imageio.ImageIO;

public class SliceImage {
	int x0Center, x1Center, yCenter;
	short[] gray, edge, mirror;
	int width, height, xMin, xMax, yMin, yMax;
	private static int border = 10;
	private final Random rnd;
	private static final Map<String, int[]> centerMemo = new HashMap<String, int[]>();

	public SliceImage(File imageFile, boolean basic) {
		rnd = new Random(imageFile.hashCode());
		BufferedImage img;
		try {
			img = ImageIO.read(imageFile);
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
		width = img.getWidth();
		height = img.getHeight();
		gray = new short[width * height];
		int[] pixels = new int[width * height];
		img.getRaster().getPixels(0, 0, img.getWidth(), img.getHeight(), pixels);
		for (int i = 0; i < pixels.length; i++) {
			int p = pixels[i] - 23;
			if (p < 1) p = 1;
			else if (p > 1279) p = 1279;
			gray[i] = (short) p;
		}
		findBody();
		int[] center = null;
		synchronized (centerMemo) {
			center = centerMemo.get(imageFile.getPath());
		}
		if (center == null) findCenterSimple();
		else {
			x0Center = center[0];
			x1Center = center[1];
			yCenter = center[2];
		}
		if (!basic) {
			buildMirror();
			buildEdge();
		}
	}

	public static void fixCenter(File imageFile, int x0, int x1, int y) {
		synchronized (centerMemo) {
			centerMemo.put(imageFile.getPath(), new int[] { x0, x1, y });
		}
	}

	private void buildEdge() {
		edge = new short[width * height];
		for (int y = 1; y < height - 1; y++) {
			int off = y * width + 1;
			for (int x = 1; x < width - 1; x++, off++) {
				// p1 p4 p6
				// p2 -- p7
				// p3 p5 p8
				int p1 = gray[off - 1 - width];
				int p2 = gray[off - 1];
				int p3 = gray[off - 1 + width];
				int p4 = gray[off - width];
				int p5 = gray[off + width];
				int p6 = gray[off + 1 - width];
				int p7 = gray[off + 1];
				int p8 = gray[off + 1 + width];
				int vert = Math.abs(p1 + 2 * p4 + p6 - p3 - 2 * p5 - p8);
				int horiz = Math.abs(p1 + 2 * p2 + p3 - p6 - 2 * p7 - p8);
				edge[off] = (short) Math.min(1279, Math.sqrt(0.49 + (vert * vert + horiz * horiz) / 2));
			}
		}
	}

	private void buildMirror() {
		int[] aux = new int[width * height];
		for (int y = 0; y < height; y++) {
			int xCenter = getXCenter(y);
			int yw = y * width;
			for (int x = 0; x < width; x++) {
				int xMirror = xCenter + xCenter - x;
				if (xMirror < 0 || xMirror >= width) continue;
				int p = x + yw;
				int v = gray[p];
				if (v == 0) continue;
				int q = gray[xMirror + yw];
				if (q == 0) continue;
				aux[p] = (short) (v - q);
			}
		}
		mirror = new short[width * height];
		for (int y = 1; y < height - 1; y++) {
			int yw = y * width;
			for (int x = 1; x < width - 1; x++) {
				int p = x + yw;
				int v = aux[p] * 4 + (aux[p - 1] + aux[p + 1] + aux[p - width] + aux[p + width]) * 2 + aux[p - 1 + width] + aux[p + 1 + width] + aux[p - 1 - width] + aux[p + 1 - width];
				mirror[p] = (short) (v / 16);
			}
		}
	}

	public int getXCenter(int y) {
		return x0Center + (x1Center - x0Center) * y / height;
	}

	private void findBody() {
		int[] queue = new int[width * height];
		int id = 1;
		int cut = 640;
		int[] group = new int[width * height];
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int p = x + y * width;
				if (gray[p] < cut) {
					group[p] = -1;
				}
			}
		}
		int largest = -1;
		int maxSize = 0;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int p = x + y * width;
				if (group[p] == 0) {
					queue[0] = p;
					int tot = 1;
					int curr = 0;
					while (curr < tot) {
						int q = queue[curr++];
						int xp = q % width;
						int yp = q / width;
						for (int i = 0; i < 4; i++) {
							int nx = i == 0 ? xp - 1 : i == 1 ? xp + 1 : xp;
							if (nx < 0 || nx >= width) continue;
							int ny = i == 2 ? yp - 1 : i == 3 ? yp + 1 : yp;
							if (ny < 0 || ny >= height) continue;
							int np = ny * width + nx;
							int na = group[np];
							if (na < 0 || na == id) continue;
							group[np] = id;
							queue[tot++] = np;
						}
					}
					if (tot > maxSize) {
						maxSize = tot;
						largest = id;
					}
					id++;
				}
			}
		}
		int tot = 0;
		for (int x = 0; x < width; x++) {
			group[tot] = id;
			queue[tot++] = x;
			group[tot] = id;
			queue[tot++] = (height - 1) * width + x;
		}
		for (int y = 0; y < width; y++) {
			group[tot] = id;
			queue[tot++] = y * width;
			group[tot] = id;
			queue[tot++] = y * width + width - 1;
		}
		int curr = 0;
		while (curr < tot) {
			int q = queue[curr++];
			int xp = q % width;
			int yp = q / width;
			for (int i = 0; i < 4; i++) {
				int nx = i == 0 ? xp - 1 : i == 1 ? xp + 1 : xp;
				if (nx < 0 || nx >= width) continue;
				int ny = i == 2 ? yp - 1 : i == 3 ? yp + 1 : yp;
				if (ny < 0 || ny >= height) continue;
				int np = ny * width + nx;
				int na = group[np];
				if (na == largest || na == id) continue;
				gray[np] = 0;
				group[np] = id;
				queue[tot++] = np;
			}
		}
		xMin = width - 1;
		xMax = 0;
		yMin = height - 1;
		yMax = 0;
		for (int y = 0; y < height; y++) {
			int yw = y * width;
			for (int x = 0; x < width; x++) {
				int v = gray[yw + x];
				if (v > 0) {
					if (x < xMin) xMin = x;
					if (x > xMax) xMax = x;
					if (y < yMin) yMin = y;
					if (y > yMax) yMax = y;
				}
			}
		}
		xMin += border;
		xMax -= border;
		yMin += border;
		yMax -= border;
	}

	private void findCenterSimple() {
		long xt = 0;
		long yt = 0;
		long div = 0;
		for (int y = 0; y < height; y++) {
			int yw = y * width;
			for (int x = 0; x < width; x++) {
				int v = gray[yw + x];
				if (v > 0) {
					xt += v * x;
					yt += v * y;
					div += v;
				}
			}
		}
		if (div > 0) {
			xt /= div;
			yt /= div;
		}
		x0Center = x1Center = (int) xt;
		yCenter = (int) yt;
	}

	public int[] findCenterMirror() {
		int x0 = x0Center;
		int x1 = x1Center;
		double min = evalMirror(x0, x1);
		for (int steps = 0; steps < 100; steps++) {
			int nx0 = x0;
			int nx1 = x1;
			int v = rnd.nextInt(3) + 1;
			if (rnd.nextInt(2) == 0) nx0 += rnd.nextInt(2) == 0 ? -v : v;
			else nx1 += rnd.nextInt(2) == 0 ? -v : v;
			double curr = evalMirror(nx0, nx1);
			if (curr < min) {
				min = curr;
				x0 = nx0;
				x1 = nx1;
			}
		}
		return new int[] { x0, x1 };
	}

	private double evalMirror(int x0, int x1) {
		double ret = 0;
		int c = 0;
		for (int y = yMin; y <= yMax; y++) {
			int yw = y * width;
			int xm = x0 + (x1 - x0) * y / height;
			for (int xa = 0; xa < xm; xa++) {
				int xb = xm + xm - xa;
				if (xb < 0 || xb >= width) continue;
				int va = gray[yw + xa];
				int vb = gray[yw + xb];
				int dif = va - vb;
				ret += dif * dif;
				c++;
			}
		}
		return c == 0 ? 0 : ret / c;
	}

	public BufferedImage getImage() {
		BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		int c = 0;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++, c++) {
				int a = gray[c] / 5;
				img.setRGB(j, i, (a << 16) + (a << 8) + a);
			}
		}
		return img;
	}

	public BufferedImage getEdgeImage() {
		BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		int c = 0;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++, c++) {
				int a = edge[c] / 5;
				img.setRGB(j, i, (a << 16) + (a << 8) + a);
			}
		}
		return img;
	}

	public BufferedImage getMirrorImage() {
		BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		int c = 0;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++, c++) {
				int a = mirror[c] / 5;
				img.setRGB(j, i, a < 0 ? ((-a) << 16) : a << 8);
			}
		}
		return img;
	}
}
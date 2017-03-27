public class LungFeatureExtractor {
	public static final int numFeatures = 1511;
	private static final int d1 = 100;
	private static final int d2 = 50;

	public float[] getFeatures(SliceImage image, Slice slice, int usedContrast) {
		float[] ret = new float[numFeatures];
		int k = 0;
		int h = image.height;
		int w = image.width;
		int[] tot = new int[15];
		int[][] count = new int[tot.length][];
		int[][] xs = new int[tot.length][];
		int[][] ys = new int[tot.length][];
		count[0] = new int[100];
		count[1] = new int[34];
		count[2] = new int[8];
		count[3] = new int[71];
		count[4] = new int[23];
		count[5] = new int[5];
		count[6] = new int[71];
		count[7] = new int[23];
		count[8] = new int[5];
		count[9] = new int[61];
		count[10] = new int[19];
		count[11] = new int[7];
		count[12] = new int[53];
		count[13] = new int[17];
		count[14] = new int[6];
		for (int i = 0; i < tot.length; i++) {
			xs[i] = new int[count[i].length];
			ys[i] = new int[count[i].length];
		}
		for (int y = 0; y < h; y++) {
			int xc = image.getXCenter(y);
			int xMin1 = Util.xToPixel(Util.pixelToX(xc, slice) - d1, slice);
			int xMax1 = Util.xToPixel(Util.pixelToX(xc, slice) + d1, slice);
			int xMin2 = Util.xToPixel(Util.pixelToX(xc, slice) - d2, slice);
			int xMax2 = Util.xToPixel(Util.pixelToX(xc, slice) + d2, slice);
			for (int x = 0; x < w; x++) {
				int v = image.gray[y * w + x] - 1;
				if (v < 0) continue;
				for (int i = 0; i < tot.length; i++) {
					if (x > xc && i >= 3 && i < 6) continue;
					if (x < xc && i >= 6 && i < 9) continue;
					if ((x < xMin1 || x > xMax1) && i >= 9 && i < 12) continue;
					if ((x < xMin2 || x > xMax2) && i >= 12 && i < 15) continue;
					int[] c = count[i];
					int div = (1280 + c.length - 1) / c.length;
					int bin = v / div;
					c[bin]++;
					xs[i][bin] += x - xc;
					ys[i][bin] += y - image.yCenter;
					tot[i]++;
				}
			}
		}
		for (int i = 0; i < tot.length; i++) {
			int[] ci = count[i];
			double ti = tot[i];
			int[] xi = xs[i];
			int[] yi = ys[i];
			for (int j = 0; j < ci.length; j++) {
				if (ti == 0) {
					k += 3;
				} else {
					ret[k++] = (float) (ci[j] / ti);
					ret[k++] = (float) (Util.pixelToX((int) Math.round(xi[j] / ti), slice));
					ret[k++] = (float) (Util.pixelToY((int) Math.round(yi[j] / ti), slice));
				}
			}
		}
		ret[k++] = (float) slice.z;
		ret[k++] = usedContrast;
		return ret;
	}
}
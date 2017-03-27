import java.util.Arrays;

public class TumorFeatureExtractor {
	public static final int numFeatures = 207;
	private static final int dark1 = 200, dark2 = 500;
	private static final int numHistBins = 12;
	private static final int numChannels = 6;
	private final int[][][] histogram;
	private final long[][][] sums0, sums1, sums2;
	private final short[][][] rawValues;
	private final int height, width, sliceZ, usedContrast, modeLeft, modeCenter, modeRight;
	private final int[] dist;
	private final float[] leftDarkPct1, leftDarkPct2, rightDarkPct1, rightDarkPct2, topDarkPct1, topDarkPct2, bottomDarkPct1, bottomDarkPct2, centerDarkPct1, centerDarkPct2;
	private final float[] leftDarkDist1, leftDarkDist2, rightDarkDist1, rightDarkDist2, topDarkDist1, topDarkDist2, bottomDarkDist1, bottomDarkDist2, centerDarkDist1, centerDarkDist2;
	private final float[] percentil;
	private final float darkPct1, darkPct2, slicePct;
	private final SliceImage image;
	private final Slice slice;

	public TumorFeatureExtractor(SliceImage imagePrev, SliceImage image, SliceImage imageNext, Slice slice, int usedContrast, double slicePct) {
		this.slice = slice;
		this.image = image;
		height = image.height;
		width = image.width;
		sliceZ = (int) slice.z;
		this.usedContrast = usedContrast;
		this.slicePct = (float) slicePct;
		leftDarkPct1 = new float[width * height];
		leftDarkPct2 = new float[width * height];
		rightDarkPct1 = new float[width * height];
		rightDarkPct2 = new float[width * height];
		topDarkPct1 = new float[width * height];
		topDarkPct2 = new float[width * height];
		bottomDarkPct1 = new float[width * height];
		bottomDarkPct2 = new float[width * height];
		centerDarkPct1 = new float[width * height];
		centerDarkPct2 = new float[width * height];
		leftDarkDist1 = new float[width * height];
		leftDarkDist2 = new float[width * height];
		rightDarkDist1 = new float[width * height];
		rightDarkDist2 = new float[width * height];
		topDarkDist1 = new float[width * height];
		topDarkDist2 = new float[width * height];
		bottomDarkDist1 = new float[width * height];
		bottomDarkDist2 = new float[width * height];
		centerDarkDist1 = new float[width * height];
		centerDarkDist2 = new float[width * height];
		dist = new int[height * width];
		Arrays.fill(dist, height + width);
		int[] q = new int[width * height * 8];
		int tot = 0;
		int c = 0;
		int darkCnt1 = 0;
		int darkCnt2 = 0;
		int totCnt = 0;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++, c++) {
				int v = image.gray[c];
				if (v == 0) {
					q[tot++] = c;
					dist[c] = 0;
				} else {
					if (v < dark1) darkCnt1++;
					if (v < dark2) darkCnt2++;
					totCnt++;
				}
			}
		}
		darkPct1 = darkCnt1 * 100.f / totCnt;
		darkPct2 = darkCnt2 * 100.f / totCnt;

		//Left		
		final int maxDist = Math.max(height, width);
		for (int y = 0; y < height; y++) {
			int cnt = 0;
			int dist1 = maxDist;
			int dist2 = maxDist;
			int pct1 = 0;
			int pct2 = 0;
			for (int x = 0; x < width; x++) {
				int p = y * width + x;
				int v = image.gray[p];
				if (v == 0) continue;
				cnt++;
				if (v < dark1) {
					dist1 = 0;
					pct1++;
				} else {
					dist1 = Math.min(maxDist, dist1 + 1);
				}
				if (v < dark2) {
					dist2 = 0;
					pct2++;
				} else {
					dist2 = Math.min(maxDist, dist2 + 1);
				}
				leftDarkDist1[p] = dist1;
				leftDarkPct1[p] = pct1 / (float) cnt;
				leftDarkDist2[p] = dist2;
				leftDarkPct2[p] = pct2 / (float) cnt;
			}
		}
		//right
		for (int y = 0; y < height; y++) {
			int cnt = 0;
			int dist1 = maxDist;
			int dist2 = maxDist;
			int pct1 = 0;
			int pct2 = 0;
			for (int x = width - 1; x >= 0; x--) {
				int p = y * width + x;
				int v = image.gray[p];
				if (v == 0) continue;
				cnt++;
				if (v < dark1) {
					dist1 = 0;
					pct1++;
				} else {
					dist1 = Math.min(maxDist, dist1 + 1);
				}
				if (v < dark2) {
					dist2 = 0;
					pct2++;
				} else {
					dist2 = Math.min(maxDist, dist2 + 1);
				}
				rightDarkDist1[p] = dist1;
				rightDarkPct1[p] = pct1 / (float) cnt;
				rightDarkDist2[p] = dist2;
				rightDarkPct2[p] = pct2 / (float) cnt;
			}
		}
		//Top		
		for (int x = 0; x < width; x++) {
			int cnt = 0;
			int dist1 = maxDist;
			int dist2 = maxDist;
			int pct1 = 0;
			int pct2 = 0;
			for (int y = 0; y < height; y++) {
				int p = y * width + x;
				int v = image.gray[p];
				if (v == 0) continue;
				cnt++;
				if (v < dark1) {
					dist1 = 0;
					pct1++;
				} else {
					dist1 = Math.min(maxDist, dist1 + 1);
				}
				if (v < dark2) {
					dist2 = 0;
					pct2++;
				} else {
					dist2 = Math.min(maxDist, dist2 + 1);
				}
				topDarkDist1[p] = dist1;
				topDarkPct1[p] = pct1 / (float) cnt;
				topDarkDist2[p] = dist2;
				topDarkPct2[p] = pct2 / (float) cnt;
			}
		}
		//Bottom		
		for (int x = 0; x < width; x++) {
			int cnt = 0;
			int dist1 = maxDist;
			int dist2 = maxDist;
			int pct1 = 0;
			int pct2 = 0;
			for (int y = height - 1; y >= 0; y--) {
				int p = y * width + x;
				int v = image.gray[p];
				if (v == 0) continue;
				cnt++;
				if (v < dark1) {
					dist1 = 0;
					pct1++;
				} else {
					dist1 = Math.min(maxDist, dist1 + 1);
				}
				if (v < dark2) {
					dist2 = 0;
					pct2++;
				} else {
					dist2 = Math.min(maxDist, dist2 + 1);
				}
				bottomDarkDist1[p] = dist1;
				bottomDarkPct1[p] = pct1 / (float) cnt;
				bottomDarkDist2[p] = dist2;
				bottomDarkPct2[p] = pct2 / (float) cnt;
			}
		}
		//Center -> Right
		for (int y = 0; y < height; y++) {
			int cnt = 0;
			int dist1 = maxDist;
			int dist2 = maxDist;
			int pct1 = 0;
			int pct2 = 0;
			int xc = image.getXCenter(y);
			for (int x = xc; x < width; x++) {
				int p = y * width + x;
				int v = image.gray[p];
				if (v == 0) continue;
				cnt++;
				if (v < dark1) {
					dist1 = 0;
					pct1++;
				} else {
					dist1 = Math.min(maxDist, dist1 + 1);
				}
				if (v < dark2) {
					dist2 = 0;
					pct2++;
				} else {
					dist2 = Math.min(maxDist, dist2 + 1);
				}
				centerDarkDist1[p] = dist1;
				centerDarkPct1[p] = pct1 / (float) cnt;
				centerDarkDist2[p] = dist2;
				centerDarkPct2[p] = pct2 / (float) cnt;
			}
		}
		//Center -> Left
		for (int y = 0; y < height; y++) {
			int cnt = 0;
			int dist1 = maxDist;
			int dist2 = maxDist;
			int pct1 = 0;
			int pct2 = 0;
			int xc = image.getXCenter(y);
			for (int x = xc - 1; x >= 0; x--) {
				int p = y * width + x;
				int v = image.gray[p];
				if (v == 0) continue;
				cnt++;
				if (v < dark1) {
					dist1 = 0;
					pct1++;
				} else {
					dist1 = Math.min(maxDist, dist1 + 1);
				}
				if (v < dark2) {
					dist2 = 0;
					pct2++;
				} else {
					dist2 = Math.min(maxDist, dist2 + 1);
				}
				centerDarkDist1[p] = dist1;
				centerDarkPct1[p] = pct1 / (float) cnt;
				centerDarkDist2[p] = dist2;
				centerDarkPct2[p] = pct2 / (float) cnt;
			}
		}

		int curr = 0;
		while (curr < tot) {
			int p = q[curr++];
			int x = p % width;
			int y = p / width;
			int nd = dist[p] + 1;
			for (int i = 0; i < 4; i++) {
				int nx = i == 0 ? x + 1 : i == 1 ? x - 1 : x;
				if (nx < 0 || nx >= width) continue;
				int ny = i == 2 ? y + 1 : i == 3 ? y - 1 : y;
				if (ny < 0 || ny >= height) continue;
				int np = ny * width + nx;
				if (nd < dist[np]) {
					dist[np] = nd;
					q[tot++] = np;
				}
			}
		}

		short[] avgChannel = new short[width * height];
		short[] difChannel = new short[width * height];
		for (int i = 0; i < avgChannel.length; i++) {
			avgChannel[i] = (short) ((image.gray[i] + imageNext.gray[i] + imagePrev.gray[i]) / 3);
			difChannel[i] = (short) ((2 * image.gray[i] - imageNext.gray[i] - imagePrev.gray[i]) / 2);
		}

		sums0 = new long[numChannels][height + 1][width + 1];
		sums1 = new long[numChannels][height + 1][width + 1];
		sums2 = new long[numChannels][height + 1][width + 1];
		rawValues = new short[numChannels][height + 1][width + 1];
		for (int channel = 0; channel < 5; channel++) {
			short[] v = null;
			if (channel == 0) v = image.gray;
			else if (channel == 1) v = image.edge;
			else if (channel == 2) v = image.mirror;
			else if (channel == 3) v = avgChannel;
			else if (channel == 4) v = difChannel;
			long sum = 0;
			long sumSquares = 0;
			long sumCubes = 0;
			short[][] rc = rawValues[channel];
			long[] scp0 = sums0[channel][0];
			long[] scp1 = sums1[channel][0];
			long[] scp2 = sums2[channel][0];
			for (int y = 0; y < height; y++) {
				sum = 0;
				sumSquares = 0;
				sumCubes = 0;
				long[] sc0 = sums0[channel][y + 1];
				long[] sc1 = sums1[channel][y + 1];
				long[] sc2 = sums2[channel][y + 1];
				int yw = y * width;
				short[] rcy = rc[y];
				for (int x = 0; x < width;) {
					short vv = rcy[x] = v[yw + x];
					x++;
					long a = vv;
					long a2 = a * a;
					sc0[x] = scp0[x] + (sum += a);
					sc1[x] = scp1[x] + (sumSquares += a2);
					sc2[x] = scp2[x] + (sumCubes += a2 * a);
				}
				scp0 = sc0;
				scp1 = sc1;
				scp2 = sc2;
			}
		}
		short[] v0 = imagePrev.gray;
		short[] v1 = image.gray;
		short[] v2 = imageNext.gray;
		int channel = 5;
		long sum = 0;
		long sumSquares = 0;
		long sumCubes = 0;
		long[] scp0 = sums0[channel][0];
		long[] scp1 = sums1[channel][0];
		long[] scp2 = sums2[channel][0];
		for (int y = 0; y < height; y++) {
			sum = 0;
			sumSquares = 0;
			sumCubes = 0;
			long[] sc0 = sums0[channel][y + 1];
			long[] sc1 = sums1[channel][y + 1];
			long[] sc2 = sums2[channel][y + 1];
			int yw = y * width;
			for (int x = 0; x < width;) {
				long a0 = v0[yw + x];
				long a1 = v1[yw + x];
				long a2 = v2[yw + x];
				x++;
				long aa0 = a0 * a0;
				long aa1 = a2 * a1;
				long aa2 = a2 * a2;
				sc0[x] = scp0[x] + (sum += a0 + a1 + a2);
				sc1[x] = scp1[x] + (sumSquares += aa0 + aa1 + aa2);
				sc2[x] = scp2[x] + (sumCubes += aa0 * a0 + aa1 * a1 + aa2 * a2);
			}
			scp0 = sc0;
			scp1 = sc1;
			scp2 = sc2;
		}

		//Histogram
		histogram = new int[height + 1][width + 1][numHistBins];
		int[][] h0 = histogram[0];
		int[] hist = new int[numHistBins];
		for (int y = 0; y < height; y++) {
			Arrays.fill(hist, 0);
			int[][] h1 = histogram[y + 1];
			int yw = y * width;
			for (int x = 0; x < width;) {
				int color = image.gray[yw + x];
				if (color > 0) hist[histBin(color)]++;
				x++;
				int[] h0x = h0[x];
				int[] h1x = h1[x];
				for (int i = 0; i < numHistBins; i++) {
					h1x[i] = h0x[i] + hist[i];
				}
			}
			h0 = h1;
		}

		int maxCnt = totCnt / 6;
		int cnt = 0;
		int[] freq = new int[1280];
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				int v = image.gray[y * width + x];
				if (v == 0) continue;
				cnt++;
				int a = Math.max(1, v - 5);
				int b = Math.min(freq.length-1, v + 5);
				for (int i = a; i <= b; i++) {
					freq[i]++;
				}
			}
			if (cnt > maxCnt) break;
		}
		int mode = 0;
		int max = 0;
		for (int i = 0; i < freq.length; i++) {
			if (freq[i] > max) {
				max = freq[i];
				mode = i;
			}
		}
		modeLeft = mode;

		cnt = 0;
		Arrays.fill(freq, 0);
		for (int x = width - 1; x >= 0; x--) {
			for (int y = 0; y < height; y++) {
				int v = image.gray[y * width + x];
				if (v == 0) continue;
				cnt++;
				int a = Math.max(1, v - 5);
				int b = Math.min(freq.length-1, v + 5);
				for (int i = a; i <= b; i++) {
					freq[i]++;
				}
			}
			if (cnt > maxCnt) break;
		}
		mode = 0;
		max = 0;
		for (int i = 0; i < freq.length; i++) {
			if (freq[i] > max) {
				max = freq[i];
				mode = i;
			}
		}
		modeRight = mode;

		cnt = 0;
		Arrays.fill(freq, 0);
		for (int dx = 0; dx < width; dx++) {
			for (int k = 0; k <= 1; k++) {
				if (dx == 0 && k != 0) continue;
				for (int y = 0; y < height; y++) {
					int x = image.getXCenter(y) + (k == 0 ? dx : -dx);
					if (x < 0 || x >= width) continue;
					int v = image.gray[y * width + x];
					if (v == 0) continue;
					cnt++;
					int a = Math.max(1, v - 5);
					int b = Math.min(freq.length-1, v + 5);
					for (int i = a; i <= b; i++) {
						freq[i]++;
					}
				}
			}
			if (cnt > maxCnt) break;
		}
		mode = 0;
		max = 0;
		for (int i = 1; i < freq.length; i++) {
			if (freq[i] > max) {
				max = freq[i];
				mode = i;
			}
		}
		modeCenter = mode;
		//System.err.println(modeCenter+":"+modeLeft+":"+modeRight);
		percentil = new float[1280];
		Arrays.fill(freq, 0);
		for (int x = width - 1; x >= 0; x--) {
			for (int y = 0; y < height; y++) {
				int v = image.gray[y * width + x];
				if (v == 0) continue;
				freq[v]++;
			}
		}
		int a = 0;
		for (int i = 1; i < percentil.length; i++) {
			a += freq[i];
			percentil[i] = a / (float) totCnt;
		}
	}

	private static int histBin(int v) {
		if (v < 700) return 0;
		if (v >= 1100) return 11;
		return (v - 700) / 40 + 1;
	}

	public float[] getFeatures(int sx, int sy) {
		float[] ret = new float[numFeatures];
		int k = 0;
		int ch = 0;
		for (int i : new int[] { 1, 2, 3, 5, 7, 9, 12, 16 }) {
			int rx = sx - i;
			int ry = sy - i;
			int rs = i * 2 + 1;
			System.arraycopy(rectStatFeatures(sums0[ch], sums1[ch], sums2[ch], rx, ry, rs, rs), 0, ret, k, 3);
			k += 3;
		}
		for (ch = 1; ch <= 2; ch++) {
			for (int i : new int[] { 1, 2, 4, 6, 8, 11, 14 }) {
				int rx = sx - i;
				int ry = sy - i;
				int rs = i * 2 + 1;
				System.arraycopy(rectStatFeatures(sums0[ch], sums1[ch], sums2[ch], rx, ry, rs, rs), 0, ret, k, 3);
				k += 3;
			}
		}
		for (ch = 3; ch <= 4; ch++) {
			for (int i : new int[] { 1, 3, 5, 7, 10, 15 }) {
				int rx = sx - i;
				int ry = sy - i;
				int rs = i * 2 + 1;
				System.arraycopy(rectStatFeatures(sums0[ch], sums1[ch], sums2[ch], rx, ry, rs, rs), 0, ret, k, 3);
				k += 3;
			}
		}
		ch = 5;
		for (int i : new int[] { 1, 2, 4, 8, 13, 17 }) {
			int rx = sx - i;
			int ry = sy - i;
			int rs = i * 2 + 1;
			System.arraycopy(rectStatFeatures(sums0[ch], sums1[ch], sums2[ch], rx, ry, rs, rs), 0, ret, k, 3);
			k += 3;
		}
		for (ch = 0; ch < 5; ch++) {
			ret[k++] = rawValues[ch][sy][sx];
		}
		for (int i : new int[] { 2, 4, 7, 11 }) {
			int rx = sx - i;
			int ry = sy - i;
			int rs = i * 2 + 1;
			System.arraycopy(rectHist(rx, ry, rs, rs), 0, ret, k, numHistBins);
			k += numHistBins;
		}

		int pos = sy * width + sx;

		float dx = ret[k++] = Util.pixelToX(sx, slice) - Util.pixelToY(image.getXCenter(sy), slice);
		float dy = ret[k++] = Util.pixelToY(sy, slice) - Util.pixelToY(image.yCenter, slice);
		ret[k++] = dx * dx + dy * dy;
		ret[k++] = Math.abs(dx);
		ret[k++] = dist[pos];
		ret[k++] = sliceZ;
		ret[k++] = darkPct1;
		ret[k++] = darkPct2;
		ret[k++] = usedContrast;
		ret[k++] = slicePct;
		ret[k++] = topDarkDist1[pos];
		ret[k++] = topDarkDist2[pos];
		ret[k++] = topDarkPct1[pos];
		ret[k++] = topDarkPct2[pos];
		ret[k++] = bottomDarkDist1[pos];
		ret[k++] = bottomDarkDist2[pos];
		ret[k++] = bottomDarkPct1[pos];
		ret[k++] = bottomDarkPct2[pos];
		ret[k++] = leftDarkDist1[pos];
		ret[k++] = leftDarkDist2[pos];
		ret[k++] = leftDarkPct1[pos];
		ret[k++] = leftDarkPct2[pos];
		ret[k++] = rightDarkDist1[pos];
		ret[k++] = rightDarkDist2[pos];
		ret[k++] = rightDarkPct1[pos];
		ret[k++] = rightDarkPct2[pos];
		ret[k++] = centerDarkDist1[pos];
		ret[k++] = centerDarkDist2[pos];
		ret[k++] = centerDarkPct1[pos];
		ret[k++] = centerDarkPct2[pos];

		int color = rawValues[0][sy][sx];
		ret[k++] = modeCenter - color;
		ret[k++] = modeLeft - color;
		ret[k++] = modeRight - color;
		ret[k++] = percentil[color];

		return ret;
	}

	private float[] rectStatFeatures(long[][] a0, long[][] a1, long[][] a2, int rx, int ry, int rw, int rh) {
		int x0 = Math.max(0, rx);
		int x1 = Math.min(width - 1, rx + rw - 1);
		int y0 = Math.max(0, ry);
		int y1 = Math.min(height - 1, ry + rh - 1);
		double sum = a0[y1 + 1][x1 + 1] - a0[y1 + 1][x0] - a0[y0][x1 + 1] + a0[y0][x0];
		double sumSquares = a1[y1 + 1][x1 + 1] - a1[y1 + 1][x0] - a1[y0][x1 + 1] + a1[y0][x0];
		double sumCubes = a2[y1 + 1][x1 + 1] - a2[y1 + 1][x0] - a2[y0][x1 + 1] + a2[y0][x0];
		int cnt = (x1 - x0 + 1) * (y1 - y0 + 1);
		float[] ret = new float[3];
		if (cnt > 0) {
			double k3 = (sumCubes - 3 * sumSquares * sum / cnt + 2 * sum * sum * sum / cnt / cnt) / cnt;
			double k2 = (sumSquares - sum * sum / cnt) / cnt;
			ret[0] = (float) (sum / cnt);
			ret[1] = (float) k2;
			ret[2] = (float) (k2 == 0 ? 0 : (k3 * k3) / (k2 * k2 * k2));
		}
		return ret;
	}

	private float[] rectHist(int rx, int ry, int rw, int rh) {
		int x0 = Math.max(0, rx);
		int x1 = Math.min(width - 1, rx + rw - 1);
		int y0 = Math.max(0, ry);
		int y1 = Math.min(height - 1, ry + rh - 1);
		int[] h0 = histogram[y1 + 1][x1 + 1];
		int[] h1 = histogram[y1 + 1][x0];
		int[] h2 = histogram[y0][x1 + 1];
		int[] h3 = histogram[y0][x0];
		float[] ret = new float[numHistBins];
		int tot = 0;
		for (int i = 0; i < numHistBins; i++) {
			tot += ret[i] = h0[i] - h1[i] - h2[i] + h3[i];
		}
		for (int i = 0; i < numHistBins; i++) {
			ret[i] /= tot;
		}
		return ret;
	}
}
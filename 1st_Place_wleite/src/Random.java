public class Random {
	private static final long mask0 = 0x80000000L;
	private static final long mask1 = 0x7fffffffL;
	private static final long[] mult = new long[] {0,0x9908b0dfL};
	private final long[] mt = new long[624];
	private int idx = 0;

	Random(long seed) {
		init(seed);
	}

	private void init(long seed) {
		mt[0] = seed & 0xffffffffl;
		for (int i = 1; i < 624; i++) {
			mt[i] = 1812433253l * (mt[i - 1] ^ (mt[i - 1] >>> 30)) + i;
			mt[i] &= 0xffffffffl;
		}
	}

	private void generate() {
		for (int i = 0; i < 227; i++) {
			long y = (mt[i] & mask0) | (mt[i + 1] & mask1);
			mt[i] = mt[i + 397] ^ (y >> 1) ^ mult[(int) (y & 1)];
		}
		for (int i = 227; i < 623; i++) {
			long y = (mt[i] & mask0) | (mt[i + 1] & mask1);
			mt[i] = mt[i - 227] ^ (y >> 1) ^ mult[(int) (y & 1)];
		}
		long y = (mt[623] & mask0) | (mt[0] & mask1);
		mt[623] = mt[396] ^ (y >> 1) ^ mult[(int) (y & 1)];
	}

	private long rand() {
		if (idx == 0) generate();
		long y = mt[idx];
		idx = (idx + 1) % 624;
		y ^= (y >> 11);
		y ^= (y << 7) & 0x9d2c5680l;
		y ^= (y << 15) & 0xefc60000l;
		return y ^ (y >> 18);
	}

	int nextInt(int n) {
		if (n <= 1) return 0;
		return (int) (rand() % n);
	}
}
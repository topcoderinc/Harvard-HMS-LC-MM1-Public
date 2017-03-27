
import java.io.File;

public class RandomForestBuilder {
	public static RandomForestPredictor train(final float[][] features, final boolean[] classif, final int totSamples, final int maxTrees, final int maxNodes, final File out, final int maxThreads,
			final int minRowsPerNode) {
		final RandomForestPredictor rf = new RandomForestPredictor(maxTrees, maxTrees * maxNodes);

		final int numThreads = Math.min(maxThreads, Runtime.getRuntime().availableProcessors());
		Thread[] threads = new Thread[numThreads];
		for (int i = 0; i < numThreads; i++) {
			final int idx = i;
			threads[i] = new Thread() {
				public void run() {
					for (int i = idx; i < maxTrees; i += numThreads) {
						ClassificationNode root = new ClassificationTree(features, classif, totSamples, i, minRowsPerNode, maxNodes).getRoot();
						synchronized (rf) {
							rf.add(root);
							if (rf.size() % 8 == 0) {
								try {
									rf.save(out);
								} catch (Exception e) {
									e.printStackTrace();
								}
							}
						}
					}
				}
			};
			threads[i].start();
			threads[i].setPriority(Thread.MIN_PRIORITY);
		}
		try {
			for (int i = 0; i < numThreads; i++) {
				threads[i].join();
			}
		} catch (InterruptedException e) {
		}
		try {
			rf.save(out);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return rf;
	}
}
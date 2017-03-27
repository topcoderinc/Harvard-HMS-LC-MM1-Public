import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class LungDetectorTrainer {
	private int numThreads = 32;
	private final int numTrees = 256;
	private int minRowsPerNode = 3;
	private int maxNodes = 16000;
	private int maxSamples = 64000;
	private int totSamples = 0;
	private Map<String,Integer> clinicalInfo; 
	private float[][] features = new float[TumorFeatureExtractor.numFeatures][maxSamples];
	private boolean[] classif = new boolean[maxSamples];

	public static void main(String[] args) {
		File trainingFolder = new File("../example");
		File rfFile = new File("model/rfLung.dat");
		File clinicalFolder = new File("../clinical");
		new LungDetectorTrainer().train(trainingFolder, rfFile, clinicalFolder);
	}

	public void train(File trainingFolder, File rfFile, File clinicalFolder) {
		List<String> patients = Util.readContent(trainingFolder);
		//patients = Util.split(patients, 0.75, true);
		clinicalInfo = Util.readClinical(clinicalFolder);
		processPatients(patients, trainingFolder);
		buildRandomForests(rfFile);
	}

	private void processPatients(final List<String> patients, final File folder) {
		try {
			System.err.println("Processing Patients");
			long t = System.currentTimeMillis();
			Thread[] threads = new Thread[numThreads];
			for (int i = 0; i < numThreads; i++) {
				final int start = i;
				threads[i] = new Thread() {
					public void run() {
						for (int j = start; j < patients.size(); j += numThreads) {
							String patient = patients.get(j);

							//////////////////////////////////////////////////
							//if (j != 0) continue;
							////////////////////////////////////////////////

							int usedContrast = clinicalInfo.get(patient);
							File[] auxFiles = Util.getAuxFiles(folder, patient);
							File[] contourFiles = new File(folder, patient + "/contours").listFiles();
							int idx = Util.findLungsStructuresIndex(new File(folder, patient + "/structures.dat"));
							String suffix = "." + idx + ".dat";
							int found = 0;
							for (File auxFile : auxFiles) {
								int p = auxFile.getName().indexOf('.');
								int sliceId = Integer.parseInt(auxFile.getName().substring(0, p));
								for (File c : contourFiles) {
									if (c.getName().startsWith(sliceId + ".") && c.getName().endsWith(suffix)) {
										found++;
										break;
									}
								}
							}
							if (found > 0) {
								for (File auxFile : auxFiles) {
									int p = auxFile.getName().indexOf('.');
									int sliceId = Integer.parseInt(auxFile.getName().substring(0, p));
									File imageFile = new File(folder, patient + "/pngs/" + sliceId + ".png");
									List<Region> regions = new ArrayList<Region>();
									Slice slice = Util.readSlice(auxFile);
									for (File c : contourFiles) {
										if (c.getName().startsWith(sliceId + ".") && c.getName().endsWith(suffix)) {
											regions.addAll(Util.extractRegions(c, slice, sliceId));
											break;
										}
									}
									SliceImage image = new SliceImage(imageFile, true);
									processImage(patient, image, regions, slice, usedContrast);
								}
							}
							System.err.println("\t\t" + patient + "\t" + (j + 1) + " : " + found + "/" + auxFiles.length);
						}
					}
				};
				threads[i].start();
				threads[i].setPriority(Thread.MIN_PRIORITY);
			}
			for (int i = 0; i < numThreads; i++) {
				threads[i].join();
			}
			System.err.println("\t         Samples: " + totSamples);
			System.err.println("\t    Elapsed Time: " + (System.currentTimeMillis() - t) + " ms");
			System.err.println();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private void processImage(String patient, SliceImage image, List<Region> regions, Slice slice, int usedContrast) {
		try {
			float[] featuresImage = new LungFeatureExtractor().getFeatures(image, slice, usedContrast);
			synchronized (classif) {
				classif[totSamples] = !regions.isEmpty();
				for (int j = 0; j < TumorFeatureExtractor.numFeatures; j++) {
					features[j][totSamples] = featuresImage[j];
				}
				totSamples++;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private void buildRandomForests(File rfFile) {
		try {
			System.err.println("Building Random Forests");
			int[] count = new int[2];
			for (int i = 0; i < totSamples; i++) {
				count[classif[i] ? 1 : 0]++;
			}

			System.err.println("==== SAMPLES =====");
			for (int i = 0; i < count.length; i++) {
				System.err.println(i + "\t" + count[i]);
			}
			System.err.println();

			long t = System.currentTimeMillis();
			if (!rfFile.getParentFile().exists()) rfFile.getParentFile().mkdirs();
			RandomForestBuilder.train(features, classif, totSamples, numTrees, maxNodes, rfFile, numThreads, minRowsPerNode);

			System.err.println("\t   RF Building: " + rfFile.length() + " bytes");
			System.err.println("\t  Elapsed Time: " + (System.currentTimeMillis() - t) + " ms");
			System.err.println();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class TumorTracerTrainer {
	private static final int numThreads = 32;
	private static final int numTrees = 128;
	private static final int minRowsPerNode = 16;
	private static final int maxNodes = 180_000;
	private static final int maxSamples = 43_000_000; 
	private static final int subSampleOut = 36;
	private static final int subSampleIn = 1;
	private Map<String, Integer> clinicalInfo;
	private int totSamples = 0;
	private float[][] features = new float[TumorFeatureExtractor.numFeatures][maxSamples];
	private boolean[] classif = new boolean[maxSamples];
	private RandomForestPredictor lungPredictor;
	private List<String> info = new ArrayList<String>();

	public static void main(String[] args) {
		File trainingFolder = new File("../example");
		File rfTumor = new File("model/rfTumor.dat");
		File rfLung = new File("model/rfLung.dat");
		File infoTumor = new File("model/infoTumor.dat");
		File clinicalFolder = new File("../clinical");
		new TumorTracerTrainer().train(trainingFolder, rfLung, rfTumor, infoTumor, clinicalFolder);
	}

	public void train(File trainingFolder, File rfLung, File rfTumor, File infoTumor, File clinicalFolder) {
		List<String> patients = Util.readContent(trainingFolder);
		//patients = Util.split(patients, 0.75, true);
		clinicalInfo = Util.readClinical(clinicalFolder);
		lungPredictor = RandomForestPredictor.loadPredictor(rfLung);
		processPatients(patients, trainingFolder);
		writeInfo(infoTumor);
		buildRandomForests(rfTumor);
	}

	private void writeInfo(File file) {
		try {
			System.err.println("Writing Info");
			long t = System.currentTimeMillis();
			BufferedWriter out = new BufferedWriter(new FileWriter(file));
			for (String s : info) {
				out.write(s);
				out.newLine();
			}
			out.close();
			System.err.println("\t          File: " + file.getPath());
			System.err.println("\t Lines Written: " + info.size());
			System.err.println("\t  Elapsed Time: " + (System.currentTimeMillis() - t) + " ms");
			System.err.println();
		} catch (Exception e) {
			e.printStackTrace();
		}
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
							//if (j > 4) continue;
							//if (!patient.equals("ANON_LUNG_TC079"))continue;
							////////////////////////////////////////////////

							int usedContrast = clinicalInfo.get(patient);
							File[] auxFiles = Util.getAuxFiles(folder, patient);
							int[] lungRange = Util.findLungRange(auxFiles, folder, patient, lungPredictor, usedContrast);
							Util.updateImagesCenter(auxFiles, folder, patient, lungRange);
							File[] contourFiles = Util.getContourFiles(folder, patient);
							int idx = Util.findTumorStructuresIndex(new File(folder, patient + "/structures.dat"));
							String suffix = "." + idx + ".dat";

							List<Region> allRegions = new ArrayList<Region>();
							SliceImage image0 = new SliceImage(new File(folder, patient + "/pngs/" + lungRange[0] + ".png"), true);
							SliceImage image1 = new SliceImage(new File(folder, patient + "/pngs/" + (lungRange[0] + 1) + ".png"), false);
							for (int sliceId = lungRange[0] + 1; sliceId < lungRange[1]; sliceId++) {
								File auxFile = new File(folder, patient + "/auxiliary/" + sliceId + ".dat");
								List<Region> regions = new ArrayList<Region>();
								Slice slice = Util.readSlice(auxFile);
								for (File c : contourFiles) {
									if (c.getName().startsWith(sliceId + ".") && c.getName().endsWith(suffix)) {
										regions.addAll(Util.extractRegions(c, slice, sliceId));
										break;
									}
								}
								processRegionsInfo(regions, slice);
								allRegions.addAll(regions);
								SliceImage image2 = new SliceImage(new File(folder, patient + "/pngs/" + (sliceId + 1) + ".png"), false);
								TumorFeatureExtractor extractor = new TumorFeatureExtractor(image0, image1, image2, slice, usedContrast, (sliceId - lungRange[0] + 1) / (double) (lungRange[1] - lungRange[0] + 1));
								processImage(patient, image1, regions, slice, extractor);
								image0 = image1;
								image1 = image2;
							}
							System.err.println("\t\t" + patient + "\t" + (j + 1) + "/" + patients.size());
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

	private void processRegionsInfo(List<Region> regions, Slice slice) {
		for (Region r : regions) {
			double xy = r.getXYProp();
			double box = r.getBoxProp();
			double min = Util.pixelToX((int) Math.round(r.getMinDim()), slice) - Util.pixelToX(0, slice);
			double max = Util.pixelToX((int) Math.round(r.getMaxDim()), slice) - Util.pixelToX(0, slice);
			String s = xy + "," + box + "," + min + "," + max;
			info.add(s);
		}
	}

	private void processImage(String patient, SliceImage image, List<Region> regions, Slice slice, TumorFeatureExtractor extractor) {
		try {
			int w = image.width;
			int h = image.height;

			BufferedImage plot = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
			Graphics2D g = plot.createGraphics();
			// g.drawImage(image.getImage(), 0, 0, null);
			g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
			g.setColor(Color.blue);
			for (Region region : regions) {
				g.fill(region.getArea());
			}
			//g.setColor(Color.red);
			//g.drawOval(image.bodyCenter.x - 1, image.bodyCenter.y - 1, 3, 3);
			g.dispose();
			// new ImgViewer(plot, patient);

			boolean[][] inside = new boolean[h][w];
			for (int y = image.yMin; y <= image.yMax; y++) {
				for (int x = image.xMin; x <= image.xMax; x++) {
					int rgb = plot.getRGB(x, y);
					inside[y][x] = (rgb & 0xff) > 128 && (((rgb >>> 8) & 255) < 16);
				}
			}

			List<float[]> featuresImage = new ArrayList<float[]>();
			List<Boolean> classifImage = new ArrayList<Boolean>();
			Random rnd = new Random(image.hashCode() + 1972);
			for (int y = image.yMin; y <= image.yMax; y++) {
				for (int x = image.xMin; x <= image.xMax; x++) {
					if (image.gray[y * w + x] == 0) continue;
					boolean in = inside[y][x];
					int subsample = in ? subSampleIn : subSampleOut;
					if (rnd.nextInt(subsample) == 0) {
						float[] arrFeatures = extractor.getFeatures(x, y);
						featuresImage.add(arrFeatures);
						classifImage.add(in);
					}
				}
			}
			synchronized (classif) {
				for (int i = 0; i < classifImage.size(); i++) {
					classif[totSamples] = classifImage.get(i).booleanValue();
					float[] v = featuresImage.get(i);
					for (int j = 0; j < TumorFeatureExtractor.numFeatures; j++) {
						features[j][totSamples] = v[j];
					}
					totSamples++;
				}
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
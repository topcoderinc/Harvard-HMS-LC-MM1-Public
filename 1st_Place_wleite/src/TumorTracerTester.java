import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Polygon;
import java.awt.RenderingHints;
import java.awt.geom.Area;
import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class TumorTracerTester {
    private static final int numThreads = 32;
    private static boolean useCache = !true;
    private final List<String> answers = new ArrayList<String>();
    private final List<String> processedPatients = new ArrayList<String>();
    private final List<Double> infoXYProp = new ArrayList<Double>();
    private final List<Double> infoBoxProp = new ArrayList<Double>();
    private final List<Double> infoMaxDim = new ArrayList<Double>();
    private final List<Double> infoMinDim = new ArrayList<Double>();
    private Map<String, Integer> clinicalInfo;
    private RandomForestPredictor tumorPredictor, lungPredictor;
    private ImgViewer viewer;
    private final long[][] evalPixelPredictions = new long[2][256];
    private static boolean eval = false;
    private static boolean view1 = !true;
    private static boolean view2 = !true;
    private static boolean write = true;
    private static final int cutLevel1 = 65;
    private static final int cutLevel2 = 20;
    private static final int cutLevel3 = 10;

    public static void main(String[] args) {
        File testingFolder = new File("../provisional");
        //testingFolder = new File("../example");
        File rfTumor = new File("model/rfTumor.dat");
        File rfLung = new File("model/rfLung.dat");
        File infoTumor = new File("model/infoTumor.dat");
        File clinicalFolder = new File("../clinical");
        File answerFile = null;
        for (int i = 1;; i++) {
            answerFile = new File("sub/res" + i + ".csv");
            if (!answerFile.exists()) break;
        }
        answerFile = new File("sub/final-" + cutLevel1 + "-" + cutLevel2 + "-" + cutLevel3 + ".csv");
        new TumorTracerTester().runTest(testingFolder, rfLung, rfTumor, answerFile, infoTumor, clinicalFolder);
    }

    public void runTest(File testingFolder, File rfLung, File rfTumor, File answerFile, File infoTumor, File clinicalFolder) {
        List<String> patients = Util.readContent(testingFolder);
        if (testingFolder.getName().indexOf("prov") < 0) {
            eval = true;
            patients = Util.split(patients, 0.75, false);
        }
        clinicalInfo = Util.readClinical(clinicalFolder);
        lungPredictor = RandomForestPredictor.loadPredictor(rfLung);
        tumorPredictor = RandomForestPredictor.loadPredictor(rfTumor);
        readInfo(infoTumor);
        processPatients(patients, testingFolder);
        if (write) writeAnswer(answerFile, processedPatients);
        if (eval) showEvalPixelPredictions();
    }

    private void showEvalPixelPredictions() {
        System.err.println();
        double err0 = 0;
        double err1 = 0;
        double div0 = 0;
        double div1 = 0;
        for (int i = 0; i < 256; i++) {
            System.err.println(evalPixelPredictions[0][i] + "\t" + evalPixelPredictions[1][i]);
            div0 += evalPixelPredictions[0][i];
            err0 += evalPixelPredictions[0][i] * i * i;
            div1 += evalPixelPredictions[1][i];
            err1 += evalPixelPredictions[1][i] * (255 - i) * (255 - i);
        }
        if (div0 > 0) err0 /= div0;
        if (div1 > 0) err1 /= div1;
        System.err.println("ERR0 = " + err0);
        System.err.println("ERR1 = " + err1);
    }

    private void processPatients(final List<String> patients, final File folder) {
        try {
            System.err.println("Processing patients");
            long t = System.currentTimeMillis();
            Thread[] threads = new Thread[numThreads];
            for (int i = 0; i < numThreads; i++) {
                final int start = i;
                threads[i] = new Thread() {
                    public void run() {
                        for (int j = start; j < patients.size(); j += numThreads) {
                            String patient = patients.get(j);

                            ///////////////////////////////////////////
                            //if (j != 9) continue;
                            //if (patient.indexOf("143") < 0) continue;
                            ///////////////////////////////////////////

                            synchronized (processedPatients) {
                                processedPatients.add(patient);
                            }
                            int usedContrast = clinicalInfo.get(patient);
                            File[] auxFiles = Util.getAuxFiles(folder, patient);
                            int[] lungRange = Util.findLungRange(auxFiles, folder, patient, lungPredictor, usedContrast);
                            Util.updateImagesCenter(auxFiles, folder, patient, lungRange);
                            File[] contourFiles = Util.getContourFiles(folder, patient);
                            int idx = Util.findTumorStructuresIndex(new File(folder, patient + "/structures.dat"));
                            String suffix = "." + idx + ".dat";

                            if (eval) {
                                for (int sliceId = 1; sliceId <= auxFiles.length; sliceId++) {
                                    if (sliceId > lungRange[0] && sliceId < lungRange[1]) continue;
                                    File auxFile = new File(folder, patient + "/auxiliary/" + sliceId + ".dat");
                                    Slice slice = Util.readSlice(auxFile);
                                    List<Region> truthRegions = new ArrayList<Region>();
                                    for (File c : contourFiles) {
                                        if (c.getName().startsWith(sliceId + ".") && c.getName().endsWith(suffix)) {
                                            truthRegions.addAll(Util.extractRegions(c, slice, sliceId));
                                            break;
                                        }
                                    }
                                    if (!truthRegions.isEmpty()) {
                                        double area = 0;
                                        for (Region r : truthRegions) {
                                            area += r.getAreaVal();
                                        }
                                        System.err.println("\t\tMISSED\t" + patient + "\t" + sliceId + "\t" + area);
                                    }
                                }
                            }
                            Map<Integer, byte[][]> valsPerSlice = new HashMap<Integer, byte[][]>();
                            List<Region> regions = new ArrayList<Region>();
                            SliceImage image0 = new SliceImage(new File(folder, patient + "/pngs/" + lungRange[0] + ".png"), true);
                            SliceImage image1 = new SliceImage(new File(folder, patient + "/pngs/" + (lungRange[0] + 1) + ".png"), false);
                            //System.err.println(lungRange[0]+":"+lungRange[1]);
                            Map<Integer, List<Region>> allTruthRegions = new HashMap<Integer, List<Region>>();
                            for (int sliceId = lungRange[0] + 1; sliceId < lungRange[1]; sliceId++) {
                                File auxFile = new File(folder, patient + "/auxiliary/" + sliceId + ".dat");
                                File imageFile = new File(folder, patient + "/pngs/" + sliceId + ".png");
                                Slice slice = Util.readSlice(auxFile);
                                List<Region> truthRegions = new ArrayList<Region>();
                                for (File c : contourFiles) {
                                    if (c.getName().startsWith(sliceId + ".") && c.getName().endsWith(suffix)) {
                                        truthRegions.addAll(Util.extractRegions(c, slice, sliceId));
                                        break;
                                    }
                                }
                                allTruthRegions.put(sliceId, truthRegions);
                                SliceImage image2 = new SliceImage(new File(folder, patient + "/pngs/" + (sliceId + 1) + ".png"), false);
                                //if (sliceId >= 34 && sliceId <= 67) {
                                List<Region> l = processImage(patient, image0, image1, image2, imageFile, slice, sliceId, usedContrast, (sliceId - lungRange[0] + 1) / (double) (lungRange[1] - lungRange[0] + 1),
                                        truthRegions, valsPerSlice);
                                updateRegionsValue(l, slice);
                                regions.addAll(l);
                                //}
                                image0 = image1;
                                image1 = image2;
                            }
                            if (regions.size() > 0) {
                                groupRegions(regions);
                                refineRegions(regions, valsPerSlice, allTruthRegions, folder, patient);
                                for (Region r : regions) {
                                    StringBuilder sb = new StringBuilder();
                                    sb.append(patient).append(',');
                                    sb.append(r.sliceId);
                                    for (int i = 0; i < r.in.npoints; i++) {
                                        Point p = Util.pixelToMm(r.in.xpoints[i], r.in.ypoints[i], r.slice);
                                        sb.append(',').append(p.x);
                                        sb.append(',').append(p.y);
                                    }
                                    synchronized (answers) {
                                        answers.add(sb.toString());
                                    }
                                }
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
            System.err.println("\t  Elapsed Time: " + (System.currentTimeMillis() - t) + " ms");
            System.err.println();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void writeAnswer(File answerFile, final List<String> patients) {
        try {
            System.err.println("Writing Answer");
            long t = System.currentTimeMillis();
            Collections.sort(answers, new Comparator<String>() {
                public int compare(String a, String b) {
                    String[] sa = a.split(",");
                    String[] sb = b.split(",");
                    int cmp = sa[0].compareTo(sb[0]);
                    if (cmp != 0) return cmp;
                    return Integer.compare(Integer.parseInt(sa[1]), Integer.parseInt(sb[1]));
                }
            });
            BufferedWriter out = new BufferedWriter(new FileWriter(answerFile), 1 << 20);
            Set<String> seen = new HashSet<String>();
            for (String s : answers) {
                seen.add(s.split(",")[0]);
                out.write(s);
                out.newLine();
            }
            for (String s : patients) {
                if (seen.contains(s)) continue;
                out.write(s + ",1,0,0,0,0,1,1,0");
                out.newLine();
            }
            out.close();
            System.err.println("\t          File: " + answerFile.getPath());
            System.err.println("\t Lines Written: " + answers.size());
            System.err.println("\t  Elapsed Time: " + (System.currentTimeMillis() - t) + " ms");
            System.err.println();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void refineRegions(List<Region> regions, Map<Integer, byte[][]> valsPerSlice, Map<Integer, List<Region>> allTruthRegions, File folder, String patient) {
        if (regions.isEmpty()) return;
        int b = 4;
        int minSliceId = regions.get(0).sliceId - b;
        int maxSliceId = regions.get(regions.size() - 1).sliceId + b;
        long ayc = 0;
        long axc = 0;
        long azc = 0;
        long agc = 0;
        long adiv = 0;
        List<Integer> lv = new ArrayList<Integer>();
        int[] mxc = new int[maxSliceId + 1];
        int[] myc = new int[maxSliceId + 1];
        for (int sliceId = minSliceId; sliceId <= maxSliceId; sliceId++) {
            SliceImage image = new SliceImage(new File(folder, patient + "/pngs/" + sliceId + ".png"), true);
            byte[][] values = valsPerSlice.get(sliceId);
            if (values == null) continue;
            int w = values[0].length;
            int h = values.length;
            BufferedImage plot = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
            Graphics2D g = plot.createGraphics();
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g.setColor(Color.blue);
            boolean found = false;
            for (Region r : regions) {
                if (sliceId == r.sliceId) {
                    found = true;
                    g.fill(r.getArea());
                }
            }
            g.dispose();
            if (found) {
                long syc = 0;
                long sxc = 0;
                long sdiv = 0;
                for (int y = image.yMin; y <= image.yMax; y++) {
                    for (int x = image.xMin; x <= image.xMax; x++) {
                        int rgb = plot.getRGB(x, y);
                        int inside = (rgb & 0xff) > 128 ? 1 : 0;
                        if (inside == 1) {
                            int val = values[y][x] & 255;

                            syc += y * val;
                            sxc += x * val;
                            sdiv += val;

                            axc += x * val;
                            ayc += y * val;
                            azc += sliceId * val;
                            agc += image.gray[y * w + x] * val;
                            adiv += val;
                            lv.add(val);
                        }
                    }
                }
                if (sdiv > 0) {
                    mxc[sliceId] = (int) (sxc / sdiv);
                    myc[sliceId] = (int) (syc / sdiv);
                }
            }
        }
        int px = 0;
        int py = 0;
        for (int sliceId = minSliceId; sliceId <= maxSliceId; sliceId++) {
            if (mxc[sliceId] == 0) {
                mxc[sliceId] = px;
                myc[sliceId] = py;
            } else {
                px = mxc[sliceId];
                py = myc[sliceId];
            }
        }
        px = py = 0;
        for (int sliceId = maxSliceId; sliceId >= minSliceId; sliceId--) {
            if (mxc[sliceId] == 0) {
                mxc[sliceId] = px;
                myc[sliceId] = py;
            } else {
                px = mxc[sliceId];
                py = myc[sliceId];
            }
        }
        Collections.sort(lv);
        int gold = lv.get(lv.size() * 3 / 4);
        //System.err.println("Gold Level = " + gold);
        if (adiv > 0) {
            axc /= adiv;
            ayc /= adiv;
            agc /= adiv;
            azc = (int) Math.round(azc / adiv);
            //System.err.println(axc + ":" + ayc + ":" + agc + ":" + azc);
        }

        List<Region> newRegions = new ArrayList<Region>();
        int[] queue = null;
        for (int sliceId = minSliceId; sliceId <= maxSliceId; sliceId++) {
            SliceImage image = new SliceImage(new File(folder, patient + "/pngs/" + sliceId + ".png"), true);
            int w = image.width;
            int h = image.height;
            byte[][] values = valsPerSlice.get(sliceId);
            if (values == null) continue;
            int xc = mxc[sliceId];
            int yc = myc[sliceId];

            if (queue == null) queue = new int[w * h];
            int[] include = new int[w * h * 2];
            int q = 12;
            double max = -1e9;
            for (int dy = yc - q; dy <= yc + q; dy++) {
                if (dy < 0 || dy >= h) continue;
                for (int dx = xc - q; dx <= xc + q; dx++) {
                    if (dx < 0 || dx >= w) continue;
                    int nv = values[dy][dx] & 255;
                    int np = dy * w + dx;
                    int dg = Math.abs(image.gray[np] - (int) agc);
                    double curr = nv - dg * 0.5 - Math.abs(yc - dy) - Math.abs(xc - dx);
                    if (curr > max) {
                        max = curr;
                        queue[0] = np;
                    }
                }
            }
            int tot = 1;
            int curr = 0;
            while (curr < tot) {
                int p = queue[curr++];
                if (include[p] == -1) continue;
                int xp = p % w;
                int yp = p / w;
                for (int i = 0; i < 4; i++) {
                    int nx = i == 0 ? xp - 1 : i == 1 ? xp + 1 : xp;
                    if (nx < 0 || nx >= w) continue;
                    int ny = i == 2 ? yp - 1 : i == 3 ? yp + 1 : yp;
                    if (ny < 0 || ny >= h) continue;
                    int np = ny * w + nx;
                    if (include[np] != 0) continue;
                    int nv = values[ny][nx] & 255;
                    int dx = nx - (int) axc;
                    int dy = ny - (int) ayc;
                    double dist = Math.sqrt(dx * dx + dy * dy);
                    int dg = Math.abs(image.gray[np] - (int) agc);
                    double mLevel = 1 - (gold - nv) / (double) (gold - cutLevel3);
                    if (mLevel < 0) mLevel = 0;
                    double mDist = 1 - dist / w * 8;
                    if (mDist < 0) mDist = 0;
                    double mColor = dg < 50 ? 1 + (50 - dg) / 50.0 : 1 - (dg - 50) / 250.0;
                    if (mColor < 0) mColor = 0;
                    double mz = 1 - Math.abs(azc - sliceId) / (double) (maxSliceId - minSliceId + 1) * 0.2;
                    if (mz < 0) mz = 0;
                    double m = mLevel * mDist * mColor * mz * 100;
                    if (nv >= gold || m > cutLevel2) {
                        include[np] = 1;
                        queue[tot++] = np;
                    } else {
                        include[np] = -1;
                    }
                }
            }
            List<Integer> l = new ArrayList<Integer>();
            for (int i = 1; i < tot; i++) {
                int p = queue[i];
                if (include[p] != 1) continue;
                int xp = p % w;
                int yp = p / w;
                boolean a1 = xp == 0 || include[p - 1] != 1;
                boolean a2 = yp == 0 || include[p - w] != 1;
                boolean a3 = xp == w - 1 || include[p + 1] != 1;
                boolean a4 = yp == h - 1 || include[p + w] != 1;
                if (a1 || a2 || a3 || a4) l.add(point(xp, yp));
            }
            File auxFile = new File(folder, patient + "/auxiliary/" + sliceId + ".dat");
            Slice slice = Util.readSlice(auxFile);
            Region newRegion = new Region(slice, sliceId);
            if (l.size() > 0) newRegion.in = SimplifyPolygon.simplify(makePolygon(l, w, h), 2);

            if (newRegion.in != null) newRegions.add(newRegion);

            if (view2) {
                List<Region> truthRegions = allTruthRegions.get(sliceId);

                BufferedImage img2 = new BufferedImage(w * 2, h * 2, BufferedImage.TYPE_INT_BGR);
                Graphics2D g = img2.createGraphics();
                g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                g.drawImage(image.getImage(), 0, 0, null);
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        int rv = values[y][x] & 255;
                        img2.setRGB(x + w, y + h, rv * (65536 + 256 + 1));
                    }
                }
                g.setColor(new Color(255, 0, 0, 150));
                g.drawLine(w / 2, 0, w / 2, h - 1);
                g.setColor(new Color(255, 255, 0, 150));
                g.drawLine(image.x0Center, 0, image.x1Center, h - 1);

                if (truthRegions != null) {
                    g.setColor(new Color(255, 0, 0, 120));
                    for (Region r : truthRegions) {
                        g.fill(r.getArea());
                    }
                    g.setColor(new Color(255, 0, 255, 150));
                    for (Region r : truthRegions) {
                        g.draw(r.getArea());
                    }
                }
                g.setColor(new Color(0, 255, 0, 120));
                for (Region r : regions) {
                    if (sliceId == r.sliceId) g.fill(r.getArea());
                }
                g.setColor(new Color(255, 255, 50, 150));
                for (Region r : regions) {
                    if (sliceId == r.sliceId) g.draw(r.getArea());
                }
                if (newRegion.in != null) {
                    g.setColor(new Color(0, 0, 250, 150));
                    g.fill(newRegion.in);
                    g.setColor(new Color(200, 200, 250, 100));
                    g.draw(newRegion.in);
                }

                g.drawImage(image.getImage(), w, 0, null);
                g.setColor(Color.YELLOW);
                g.drawOval((int) xc - 1, (int) yc - 1, 3, 3);

                g.dispose();
                if (viewer == null) viewer = new ImgViewer(img2, String.valueOf(sliceId));
                else viewer.add(img2, String.valueOf(sliceId));
            }
        }
        if (!newRegions.isEmpty()) {
            regions.clear();
            regions.addAll(newRegions);
        }
    }

    private List<Region> processImage(String patient, SliceImage imagePrev, SliceImage image, SliceImage imageNext, File imageFile, Slice slice, int sliceId, int usedContrast, double slicePct, List<Region> truthRegions,
            Map<Integer, byte[][]> valsPerSlice) {
        List<Region> regions = new ArrayList<Region>();
        try {
            int w = image.width;
            int h = image.height;
            byte[][] values = null;
            //long t = System.currentTimeMillis();
            File f = new File(imageFile.getAbsolutePath() + ".cache");

            if (f.exists() && useCache) {
                values = new byte[h][w];
                BufferedInputStream in = new BufferedInputStream(new FileInputStream(f), w * h);
                for (int j = 0; j < h; j++) {
                    in.read(values[j]);
                }
                in.close();
            } else {
                values = Util.evalImage(imagePrev, image, imageNext, slice, tumorPredictor, usedContrast, slicePct);
                if (useCache) {
                    BufferedOutputStream out = new BufferedOutputStream(new FileOutputStream(f), w * h);
                    for (int j = 0; j < h; j++) {
                        out.write(values[j]);
                    }
                    out.close();
                }
            }
            byte[][] sv = new byte[h][w];
            for (int y = image.yMin; y <= image.yMax; y++) {
                byte[] r0 = values[y - 1];
                byte[] ry = values[y];
                byte[] r1 = values[y + 1];
                for (int x = image.xMin; x <= image.xMax; x++) {
                    if (image.gray[y * w + x] == 0) continue;
                    sv[y][x] = (byte) ((4 * (ry[x] & 255) + (r0[x - 1] & 255) + (r0[x + 1] & 255) + (r1[x - 1] & 255) + (r1[x + 1] & 255) + ((r0[x] & 255) + (r1[x] & 255) + (ry[x - 1] & 255) + (ry[x + 1] & 255)) * 2)
                            / 16);
                }
            }
            values = sv;
            valsPerSlice.put(sliceId, values);

            if (eval) {
                BufferedImage plot = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
                Graphics2D g = plot.createGraphics();
                g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                g.setColor(Color.blue);
                for (Region region : truthRegions) {
                    g.fill(region.getArea());
                }
                g.dispose();
                int zeros = 0;
                long[][] imageEvalPixelPredictions = new long[2][256];
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        boolean empty = image.gray[y * w + x] == 0;
                        int rgb = plot.getRGB(x, y);
                        int inside = (rgb & 0xff) > 128 ? 1 : 0;
                        int val = values[y][x] & 255;
                        if (!empty) imageEvalPixelPredictions[inside][val]++;
                        if (inside == 1 && val == 0 && empty) zeros++;
                    }
                }
                synchronized (evalPixelPredictions) {
                    for (int i = 0; i < 256; i++) {
                        evalPixelPredictions[0][i] += imageEvalPixelPredictions[0][i];
                        evalPixelPredictions[1][i] += imageEvalPixelPredictions[1][i];
                    }
                }
                if (zeros > 0) System.err.println("ZEROS\t" + patient + "\t" + sliceId + "\t" + zeros);
            }
            regions.addAll(findRegions(image, values, cutLevel1, 10, 10, 3, slice, sliceId, truthRegions));//AAA
            //			System.err.println("ProcessImage: " + (System.currentTimeMillis() - t));

        } catch (Exception e) {
            e.printStackTrace();
        }
        return regions;
    }

    private List<Region> findRegions(SliceImage image, byte[][] values, int cut, int minPoints, double minArea, int simplify, Slice slice, int sliceId, List<Region> truthRegions) {
        List<Region> regions = new ArrayList<Region>();
        int h = values.length;
        int w = values[0].length;
        int[][] a = new int[h][w];
        for (int y = image.yMin; y <= image.yMax; y++) {
            byte[] vy = values[y];
            int[] ay = a[y];
            for (int x = image.xMin; x <= image.xMax; x++) {
                if (image.gray[y * w + x] == 0) continue;
                if ((vy[x] & 255) > cut) ay[x] = 1;
            }
        }

        int[] queue = new int[w * h];
        int id = 1;
        List<Integer> l = new ArrayList<Integer>();
        for (int y = image.yMin; y <= image.yMax; y++) {
            int[] ay = a[y];
            for (int x = image.xMin; x <= image.xMax; x++) {
                if (ay[x] == 1) {
                    ay[x] = ++id;
                    queue[0] = point(x, y);
                    int tot = 1;
                    int curr = 0;
                    int sum = values[y][x] & 255;
                    while (curr < tot) {
                        int p = queue[curr++];
                        int xp = x(p);
                        int yp = y(p);
                        for (int i = 0; i < 4; i++) {
                            int nx = i == 0 ? xp - 1 : i == 1 ? xp + 1 : xp;
                            if (nx < 0 || nx >= w) continue;
                            int ny = i == 2 ? yp - 1 : i == 3 ? yp + 1 : yp;
                            if (ny < 0 || ny >= h) continue;
                            int na = a[ny][nx];
                            if (na == 0 || na == id) continue;
                            a[ny][nx] = id;
                            sum += values[ny][nx] & 255;
                            queue[tot++] = point(nx, ny);
                        }
                    }
                    if (tot < minPoints) continue;

                    l.clear();
                    for (int i = 0; i < tot; i++) {
                        int p = queue[i];
                        int xp = x(p);
                        int yp = y(p);
                        boolean a1 = xp == 0 || a[yp][xp - 1] != id;
                        boolean a2 = yp == 0 || a[yp - 1][xp] != id;
                        boolean a3 = xp == w - 1 || a[yp][xp + 1] != id;
                        boolean a4 = yp == h - 1 || a[yp + 1][xp] != id;
                        if (a1 || a2 || a3 || a4) l.add(p);
                    }
                    // System.err.println(x + ":" + y + ":" + tot + ":" + l.size());
                    Region r = new Region(slice, sliceId);
                    r.in = makePolygon(l, w, h);
                    r.value = sum / tot; ///AAAA
                    if (r.in != null && r.getAreaVal() > minArea) {
                        Polygon sp = SimplifyPolygon.simplify(r.in, simplify);
                        r.in = sp;
                        if (r.in != null) {
                            r.invalidate();
                            if (r.in.npoints > 2 && r.getAreaVal() > minArea) regions.add(r);
                        }
                    }
                }
            }
        }
        Collections.sort(regions, new Comparator<Region>() {
            public int compare(Region a, Region b) {
                return Double.compare(b.value, a.value);
            }
        });
        if (regions.size() > 10) regions.subList(10, regions.size()).clear();

        if (view1) {
            BufferedImage img2 = new BufferedImage(w * 2, h, BufferedImage.TYPE_INT_BGR);
            Graphics2D g = img2.createGraphics();
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g.drawImage(image.getImage(), 0, 0, null);
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int rv = values[y][x] & 255;
                    img2.setRGB(x + w, y, rv * (65536 + 256 + 1));
                }
            }
            g.setColor(new Color(255, 0, 0, 150));
            g.drawLine(w / 2, 0, w / 2, h - 1);
            g.setColor(new Color(255, 255, 0, 150));
            g.drawLine(image.x0Center, 0, image.x1Center, h - 1);

            g.setColor(new Color(255, 0, 0, 120));
            for (Region r : truthRegions) {
                g.fill(r.getArea());
            }
            g.setColor(new Color(255, 0, 255, 150));
            for (Region r : truthRegions) {
                g.draw(r.getArea());
            }
            g.setColor(new Color(0, 255, 0, 120));
            for (Region r : regions) {
                g.fill(r.getArea());
            }
            g.setColor(new Color(255, 255, 50, 150));
            for (Region r : regions) {
                g.draw(r.getArea());
            }
            //g.drawImage(image.getMirrorImage(), w, 0, null);

            g.dispose();
            if (viewer == null) viewer = new ImgViewer(img2, String.valueOf(sliceId));
            else viewer.add(img2, String.valueOf(sliceId));
        }

        return regions;
    }

    private static Polygon makePolygon(List<Integer> p, int w, int h) {
        Set<Integer> free = new HashSet<Integer>(p);
        for (int curr : p) {
            for (int dy = 0; dy <= 1; dy++) {
                for (int dx = 0; dx <= 1; dx++) {
                    if (dy == 0 && dx == 0) continue;
                    int nx = x(curr) + dx;
                    int ny = y(curr) + dy;
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                    free.add(point(nx, ny));
                }
            }
        }
        if (p.size() < 3) return null;
        int base = p.get(0);
        for (int curr : free) {
            if (y(curr) < y(base) || (y(curr) == y(base) && x(curr) > x(base))) {
                base = curr;
            }
        }
        List<Integer> seq = new ArrayList<Integer>();
        List<Integer> sd = new ArrayList<Integer>();
        seq.add(base);
        // 3
        // |
        // 0---+---2
        // |
        // 1
        int dir = 0;
        int curr = base;
        int[] o0 = new int[] {3,0,1};
        int[] o1 = new int[] {0,1,2};
        int[] o2 = new int[] {1,2,3};
        int[] o3 = new int[] {2,3,0};
        int[][] o = new int[][] {o0,o1,o2,o3};
        while (true) {
            int x = x(curr);
            int y = y(curr);
            int[] ord = o[dir];
            int next = -1;
            int nd = -1;
            for (int i : ord) {
                int nx = x;
                int ny = y;
                if (i == 0) nx--;
                else if (i == 1) ny++;
                else if (i == 2) nx++;
                else if (i == 3) ny--;
                if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                int np = point(nx, ny);
                if (free.remove(np)) {
                    next = np;
                    nd = i;
                    break;
                }
            }
            if (next == -1) {
                if (curr == base) break;
                seq.remove(seq.size() - 1);
                curr = seq.get(seq.size() - 1);
                dir = sd.remove(sd.size() - 1);
            } else {
                if (next == base) break;
                seq.add(next);
                sd.add(nd);
                dir = nd;
                curr = next;
            }
        }
        if (seq.size() < 3) return null;
        for (int i = 1; i < seq.size() - 1; i++) {
            int a = seq.get(i - 1);
            int b = seq.get(i);
            int c = seq.get(i + 1);
            if ((x(a) == x(b) && x(b) == x(c)) || (y(a) == y(b) && y(b) == y(c))) {
                seq.remove(i--);
            }
        }
        int[] xp = new int[seq.size()];
        int[] yp = new int[seq.size()];
        for (int i = 0; i < xp.length; i++) {
            int v = seq.get(i);
            xp[i] = x(v);
            yp[i] = y(v);
        }
        Polygon poly = new Polygon(xp, yp, xp.length);
        return poly;
    }

    private static final int point(int x, int y) {
        return (y << 16) | x;
    }

    private static final int x(int p) {
        return p & 0xffff;
    }

    private static final int y(int p) {
        return p >>> 16;
    }

    private void readInfo(File file) {
        try {
            BufferedReader in = new BufferedReader(new FileReader(file));
            String s = null;
            while ((s = in.readLine()) != null) {
                String[] v = s.split(",");
                infoXYProp.add(Double.parseDouble(v[0]));
                infoBoxProp.add(Double.parseDouble(v[1]));
                infoMinDim.add(Double.parseDouble(v[2]));
                infoMaxDim.add(Double.parseDouble(v[3]));
            }
            in.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        Collections.sort(infoXYProp);
        Collections.sort(infoBoxProp);
        Collections.sort(infoMinDim);
        Collections.sort(infoMaxDim);
    }

    public void updateRegionsValue(List<Region> regions, Slice slice) {
        for (int i = 0; i < regions.size(); i++) {
            Region r = regions.get(i);
            r.value *= evalInfo(r.getXYProp(), infoXYProp, true, false);
            r.value *= evalInfo(r.getBoxProp(), infoBoxProp, true, true);
            double min = Util.pixelToX((int) Math.round(r.getMinDim()), slice) - Util.pixelToX(0, slice);
            double max = Util.pixelToX((int) Math.round(r.getMaxDim()), slice) - Util.pixelToX(0, slice);
            r.value *= evalInfo(min, infoMinDim, true, false);
            r.value *= evalInfo(max, infoMaxDim, false, true);
            if (r.value < 0.01) regions.remove(i--);///AAA
        }
    }

    private double evalInfo(double v, List<Double> l, boolean min, boolean max) {
        int p = Collections.binarySearch(l, v);
        if (p < 0) p = -p - 1;
        for (int i = 0; i < 2; i++) {
            if (i == 0 && !min) continue;
            if (i == 1 && !max) continue;
            if (i == 1) p = l.size() - p;
            double a = p / (double) l.size();
            if (a < 0.10) return a * a * 100;
        }
        return 1;
    }

    private static void groupRegions(List<Region> regions) {
        List<List<Region>> groups = new ArrayList<List<Region>>();
        for (Region r : regions) {
            double max = 0.3;
            List<Region> ins = null;
            for (List<Region> group : groups) {
                Region a = group.get(group.size() - 1);
                if (r.sliceId - a.sliceId > 1) continue;
                double curr = iou(r, a);
                if (curr > max) {
                    max = curr;
                    ins = group;
                }
            }
            if (ins == null) groups.add(ins = new ArrayList<Region>());
            ins.add(r);
        }
        final Map<List<Region>, Double> groupVal = new HashMap<List<Region>, Double>();
        for (int i = 0; i < groups.size(); i++) {
            List<Region> group = groups.get(i);
            double val = 0;
            for (Region r : group) {
                val = Math.max(r.value, val);
            }
            groupVal.put(group, val);
        }
        Collections.sort(groups, new Comparator<List<Region>>() {
            public int compare(List<Region> a, List<Region> b) {
                return Double.compare(groupVal.get(b), groupVal.get(a));
            }
        });
        regions.clear();
        if (!groups.isEmpty()) regions.addAll(groups.get(0));
    }

    private static double iou(Region a, Region b) {
        Area iArea = new Area(a.getArea());
        iArea.intersect(b.getArea());
        double inter = Util.areaVal(iArea);
        double div = a.getAreaVal() + b.getAreaVal() - inter;
        if (div <= 0) return 0;
        return inter / div;
    }
}
package tools;

import java.awt.Transparency;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.ComponentColorModel;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferUShort;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.FileReader;
import java.io.LineNumberReader;

import javax.imageio.ImageIO;

public class ImageConverter {
	private static final int w = 512;
	
	private void convert(File input, File outDir) throws Exception {
		if (input.isDirectory()) {
			for (File f: input.listFiles()) {
				convertFile(f, outDir);
			}
		}
		else {
			convertFile(input, outDir);
		}
	}
	
	private void convertFile(File input, File outDir) throws Exception {
		String name = input.getName();
		if (!name.endsWith(".image")) return;
		name = name.replace(".image", "");
		System.out.println("Processing " + name);
		
		int max = 0;
		int[] raw = new int[w*w];
		LineNumberReader lnr = new LineNumberReader(new FileReader(input));
		int j = 0;
		while (true) {
			String line = lnr.readLine();
			if (line == null || j >= w) break;
			String[] parts = line.split(",");
			for (int i = 0; i < parts.length; i++) {
				if (i >= w) break;
				int c = Integer.parseInt(parts[i]);
				raw[j*w + i] = c;
				max = Math.max(max, c);
			}
			j++;
		}
		lnr.close();
		System.out.println(max);
		
		short[] pixels = new short[w*w];
		short[] scaledPixels = new short[w*w];
		if (max > 0) {
			double ratio = (double)65535 / max;
			for (int i = 0; i < raw.length; i++) {
				pixels[i] = (short)raw[i];
				int scaled = (int)(ratio * raw[i]);
				scaledPixels[i] = (short)(scaled);
			}
		}
		write(new File(outDir, name + ".png"), pixels);
		//write(new File(outDir, name + "-scaled.png"), scaledPixels);
	}
	
	private static void write(File out, short[] pixels) throws Exception {
		ColorModel colorModel = new ComponentColorModel(
	            ColorSpace.getInstance(ColorSpace.CS_GRAY),
	            new int[]{16}, false, false, Transparency.OPAQUE, DataBuffer.TYPE_USHORT);
	    DataBufferUShort db = new DataBufferUShort(pixels, pixels.length);
	    WritableRaster raster = Raster.createInterleavedRaster(db, w, w, w, 1, new int[1], null);
	    BufferedImage img = new BufferedImage(colorModel, raster, false, null);
	    ImageIO.write(img, "png", out);
	}
	
	public static void main(String[] a) throws Exception {
		ImageConverter ic = new ImageConverter();
		File input = new File("../data/extraction_example/images/");//ANON_LUNG_TC001.100.image");
		File outDir = new File("../out");
		ic.convert(input, outDir);
	}	
}

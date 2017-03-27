
import java.awt.image.*;
import java.io.*;
import java.util.*;
import javax.imageio.ImageIO;

public class TumorFinder{
  final int maxNeck = 60, lowCut = 550, highCut = 1200;
  boolean testShapes = false;
  
  // points defining polygons within the lung area
  int[][][] lungPoints = {
    // 1rst range
    { {0,50,  8,20,  25,0,  28,40,  35,50,  35,70,  25,100,  20,100,
       4,80},
      {100,50,  92,20,  75,0,  70,55,  63,80,  75,100,  80,100,  
       96,80} 
    },
    // 2nd range
    { {0,50,  8,20,  25,0,  30,5,  30,30,  27,50,  40,60,  35,85,
       30,100,  20,100,  5,80},
      {100,50,  92,20,  75,0,  62,10,  78,56,  65,75,  60,80,  65,95,
       85,100,  95,80} 
    },
    // 3rd range
    { {0,50,  8,20,  25,0,  40,5,  30,30,  25,60,  40,68,  35,85,
       30,100,  20,100,  5,80},
      {100,50,  92,20,  75,0,  60,5,  80,60,  70,75,  60,80,  65,95,
       85,100,  95,80} 
    },
    // 4th range
    { {0,50,  8,20,  25,0,  40,3,  32,25,  25,50,  25,60,  40,70,
       30,95,  20,95,  5,80},
      {100,50,  90,20,  84,15,  75,20,  80,55,  62,85,  62,95,
       80,100,  95,80} 
    },
    // 5th range
    { {0,50,  40,65,  36,90,  28,100,  20,100,  5,80},
      {100,50,  94,45, 88,50,  65,80,  60,90, 70,100,  80,100,  95,80} 
    }
  };
  
  // sourceDir: input directory for lung data files
  // solutionPath: output file path containing tumor location predictions
  void run(String sourceDir, String solutionPath){
    try(PrintWriter solution = new PrintWriter(solutionPath)){
      Data sourceData = new Data(sourceDir, null);
      sourceData.loadMetaData();
      
      // read through all scans in sourceDir
      for (String scanId : sourceData.scanIds){
        // read through slices to try to get z value of neck 
        // and y bounds of body center
        Scan scan = sourceData.idToScan.get(scanId);
        int nSlice = scan.slices.size(), maxY1Body = 0, kNeck = 1, 
            imageW = 512, imageH = 512;
        int[] x1Lung = new int[nSlice+1], y1Lung = new int[nSlice+1],
              x2Lung = new int[nSlice+1], y2Lung = new int[nSlice+1];
        int[] nLowLeft = new int[nSlice+1], nLowRight = new int[nSlice+1];
        
        // image data
        short[][][] data = new short[nSlice+1][][];
             
        // determine y value of front and back of person
        for (int k = 1; k <= nSlice; k++){
          BufferedImage image = loadImage(sourceDir, scanId, k);
          DataBuffer buf = image.getRaster().getDataBuffer();
          imageW = image.getWidth(); imageH = image.getHeight();
          data[k] = new short[imageH][imageW];
          for (int j = 0; j < imageH; j++)
            for (int i = 0; i < imageW;i++)
              data[k][j][i] = getC(buf, imageW*j + i);
          
          int y1Body, y2Body;
          for (y1Body = 0; y1Body < imageH; y1Body++)
            if (data[k][y1Body][imageW/2] > lowCut &&
                data[k][y1Body+5][imageW/2] > lowCut &&
                data[k][y1Body+10][imageW/2] > lowCut) break;
          
          if (y1Body > maxY1Body) maxY1Body = y1Body;
          if (maxY1Body-y1Body < 6) kNeck = k;
          
          for (y2Body = imageH*9/10; y2Body >= 10; y2Body--)
            if ((data[k][y2Body][imageW/2-20] > 800) && 
                (data[k][y2Body-5][imageW/2-20] > lowCut) &&
                (data[k][y2Body-10][imageW/2-20] > lowCut) && 
                (data[k][y2Body][imageW/2+20] > 800) && 
                (data[k][y2Body-5][imageW/2+20] > lowCut) &&
                (data[k][y2Body-10][imageW/2+20] > lowCut)) break; 
          
          int yCenter = (y1Body + y2Body)/2, dyBody = y2Body - y1Body;
          
          // determine x value of left and right boundary of person
          int x1Body = 20, x2Body = imageW-1-20;
          for (int i = 0; i < imageW/2; i++)
            if (data[k][yCenter-dyBody/4][i] > lowCut &&
                data[k][yCenter-dyBody/4][i+5] > lowCut &&
                data[k][yCenter-dyBody/4][i+10] > lowCut &&
                data[k][yCenter+dyBody/4][i] > lowCut &&
                data[k][yCenter+dyBody/4][i+5] > lowCut &&
                data[k][yCenter+dyBody/4][i+10] > lowCut){ x1Body = i; break; }
          for (int i = imageW-1; i > imageW/2; i--)
            if (data[k][yCenter-dyBody/4][i] > lowCut &&
                data[k][yCenter-dyBody/4][i-5] > lowCut &&
                data[k][yCenter-dyBody/4][i-10] > lowCut &&
                data[k][yCenter+dyBody/4][i] > lowCut &&
                data[k][yCenter+dyBody/4][i-5] > lowCut &&
                data[k][yCenter+dyBody/4][i-10] > lowCut){ x2Body = i; break; }
          
          // determine lung region
          int dxBody = x2Body-x1Body, xCenter = (x1Body + x2Body)/2;
          x1Lung[k] = x1Body + dxBody/20; x2Lung[k] = x2Body - dxBody/20;
          y1Lung[k] = y1Body + dyBody/10; y2Lung[k] = y2Body - dyBody/10;
          
          while(x1Lung[k] < xCenter){
            boolean done = false;
            for (int j = yCenter-dyBody/4; j < yCenter+dyBody/4; j++)
              if (data[k][j][x1Lung[k]] < lowCut){ done = true; break; }
            if (done) break;
            x1Lung[k]++;
          }
          while(x2Lung[k] > xCenter){
            boolean done = false;
            for (int j = yCenter-dyBody/4; j < yCenter+dyBody/4; j++)
              if (data[k][j][x2Lung[k]] < lowCut){ done = true; break; }
            if (done) break;
            x2Lung[k]--;
          }
          int dxLung = x2Lung[k]-x1Lung[k];
          while(y1Lung[k] < yCenter){
            boolean done = false;
            for (int i = xCenter - dxLung/4; i < xCenter - dxLung/8; i++)
              if (data[k][y1Lung[k]][i] < lowCut){ done = true; break; }
            for (int i = xCenter + dxLung/8; i < xCenter + dxLung/4; i++)
              if (data[k][y1Lung[k]][i] < lowCut){ done = true; break; }
            if (done) break;
            y1Lung[k]++;
          }
          while(y2Lung[k] > yCenter){
            boolean done = false;
            for (int i = xCenter - dxLung/3; i < xCenter - dxLung/8; i++)
              if (data[k][y2Lung[k]][i] < lowCut){ done = true; break; }
            for (int i = xCenter + dxLung/8; i < xCenter + dxLung/3; i++)
              if (data[k][y2Lung[k]][i] < lowCut){ done = true; break; }
            if (done) break;
            y2Lung[k]--;
          }
        }
        kNeck = Math.min(kNeck, maxNeck);  // base of neck slice 
        
        // try to find range of slices containing lungs: k1 to k2
        int firstK = 1;
        for (int k = Math.min(kNeck+20, nSlice); k >= 1; k--){
          if (Math.abs(x2Lung[k]-imageW/2) < 20 || 
              Math.abs(x1Lung[k]-imageW/2) < 20 ||
              Math.abs(y2Lung[k]-y1Lung[k]) < 20){
            firstK = k+1; break;
          }
        }
        
        for (int k = 1; k < firstK; k++)
          x1Lung[k] = x2Lung[k] = y1Lung[k] = y2Lung[k] = 0;
        
        
        // fix left or right side of lung not being identified
        for (int k = 1; k <= nSlice; k++){
          int w1 = imageW/2 - x1Lung[k], w2 = x2Lung[k]-imageW/2;
          if (w1 > w2*3/2) x2Lung[k] = imageW/2+w1;
          else if (w2 > w1*3/2) x1Lung[k] = imageW/2-w1;
        }
        
        // get number of dark pixels in lung area
        for (int k = 1; k <= nSlice; k++){
          for (int j = y1Lung[k]; j < y2Lung[k]; j++)
            for (int i = x1Lung[k]; i < x2Lung[k]; i++)
              if (data[k][j][i] < lowCut)
                if (i < imageW/2) nLowLeft[k]++; else nLowRight[k]++;
        }
        
        // use slice range: k1 <= used slices < k2
        int k1 = 0, k2 = 0, maxLow = 0;
        for (int k = 1; k < nSlice+1; k++){
          if (k1 == 0 && (nLowLeft[k] > 20 || nLowRight[k] > 20)) k1 = k;
          int nLow = nLowLeft[k]+nLowRight[k];
          if (nLow > maxLow) maxLow = nLow;
          if (k1 != 0 && k-k1 > 30 && nLow < maxLow/2){ k2 = k-8; break; }
        }
        if (k2 == 0) k2 = nSlice+1;
        System.out.println(scanId + "   k1: " + k1 + " / k2: " + k2);
        
        // grid size = w x h / 4x4 sized squares
        int w = imageW/4, h = imageH/4;
        boolean[][][] keep = new boolean[k2][h][w],
                      mid = new boolean[k2][h][w];
        boolean[][] low = new boolean[h][w],
                    high = new boolean[h][w];
        int[][][] used = new int[k2][h][w];  // contains cluster index 
        int[][] score = new int[k2][], // score for each cluster
                nSquares = new int[k2][]; // total squares in each cluster
        
        int dk = k2-k1;
        int slicesCut1 = dk/5, slicesCut2 = dk*2/5, 
            slicesCut3 = dk*3/5, slicesCut4 = dk*4/5;
        
        // find all clusters of squares for each slice
        for (int k = k1; k < k2; k++){
          Polygon[][] lung = new Polygon[lungPoints.length][2];
          for (int i = 0; i < lung.length; i++)
            for (int j = 0; j < 2; j++)
              lung[i][j] = createLungPoly(lungPoints[i][j],
                   x1Lung[k], y1Lung[k], x2Lung[k], y2Lung[k]);
          
          // define grid values using pixels inside each grid square
          for (int j = 0; j < h; j++){
            for (int i = 0; i < w; i++){
              int nLow = 0, nHigh = 0;
              for (int jj = 0; jj < 4; jj++)
                for (int ii = 0; ii < 4; ii++){
                  short c = data[k][j*4 + jj][i*4 + ii];
                  if (c < lowCut) nLow++;
                  if (c > highCut) nHigh++;
                }
              low[j][i] = nLow >= 1;
              high[j][i] = nHigh >= 1;
              mid[k][j][i] = !low[j][i] && !high[j][i];
              
              if (testShapes) keep[k][j][i] = true;
              else keep[k][j][i] = !low[j][i];
            }
          }
        
          // keep only squares within the lung shape
          for (int j = 0; j < h; j++)
            for (int i = 0; i < w/2; i++)
              if (k-k1 < slicesCut1) 
                keep[k][j][i]&= lung[0][0].shape.contains(i*4, j*4);
              else if (k-k1 < slicesCut2) 
                keep[k][j][i]&= lung[1][0].shape.contains(i*4, j*4);
              else if (k-k1 < slicesCut3) 
                keep[k][j][i]&= lung[2][0].shape.contains(i*4, j*4);
              else if (k-k1 < slicesCut4) 
                keep[k][j][i]&= lung[3][0].shape.contains(i*4, j*4);
              else
                keep[k][j][i]&= lung[4][0].shape.contains(i*4, j*4);
          
          for (int j = 0; j < h; j++)
            for (int i = w/2; i < w; i++)
              if (k-k1 < slicesCut1) 
                keep[k][j][i]&= lung[0][1].shape.contains(i*4, j*4);
              else if (k-k1 < slicesCut2) 
                keep[k][j][i]&= lung[1][1].shape.contains(i*4, j*4);
              else if (k-k1 < slicesCut3) 
                keep[k][j][i]&= lung[2][1].shape.contains(i*4, j*4);
              else if (k-k1 < slicesCut4) 
                keep[k][j][i]&= lung[3][1].shape.contains(i*4, j*4);
              else
                keep[k][j][i]&= lung[4][1].shape.contains(i*4, j*4);
          
          if (!testShapes){
          // assume bone if high
          for (int j = 1; j < h-1; j++)
            for (int i = 1; i < w-1; i++)
              if (high[j][i])
                for (int jj = j-1; jj <= j+1; jj++)
                  for (int ii = i-1; ii <= i+1; ii++)
                    keep[k][jj][ii] = false;
          
          // prevent 2 groups from being joined by a 1 square wide bridge
          for (int j = 1; j < h-1; j++)
            for (int i = 1; i < w-1; i++){
              if (keep[k][j][i]){
                if ( (!keep[k][j][i-1] && !keep[k][j][i+1] && 
                      keep[k][j-1][i] && keep[k][j+1][i]) ||
                     (!keep[k][j-1][i] && !keep[k][j+1][i] &&
                      keep[k][j][i-1] && keep[k][j][i+1]) )
                         keep[k][j][i] = false;
              }
            }
          }
          
          // find all clusters of kept squares
          int nClusters = 1;  // total clusters is nClusters-1
          for (int j = 1; j < h; j++)
            for (int i = 1; i < w; i++)
              nClusters = fillGroup(i, j, keep[k], used[k], nClusters);
          
          nSquares[k] = new int[nClusters];
          int[] nGood = new int[nClusters], nBad = new int[nClusters];
          for (int j = 0; j < h; j++)
            for (int i = 1; i < w-1; i++){
              int g = used[k][j][i];
              if (g == 0) continue;
              if (low[j][i-1] || low[j][i+1] ||
                  low[j-1][i] || low[j+1][i]) nGood[g]++;
              if ((!low[j][i-1] && !keep[k][j][i-1]) ||
                  (!low[j][i+1] && !keep[k][j][i+1]) ||
                  (!low[j-1][i] && !keep[k][j-1][i]) ||
                  (!low[j+1][i] && !keep[k][j+1][i])) nBad[g]++;
              nSquares[k][g]++;
            }
          
          // calculate a score for each cluster
          score[k] = new int[nClusters];
          for (int i = 1; i < nSquares[k].length; i++)
            score[k][i] = 
                 (int)(Math.sqrt(nSquares[k][i])*100*(nGood[i]+1)/(nBad[i]+1));
        }
        
        // groups of clusters across multiple slices
        int[] bestGroup = new int[k2],
              currentGroup = new int[k2];  // 0 if no cluster in slice
        int bestScore = 0, currentScore;
        final int minScore = 200;
        
        for (int k = k1; k < k2; k++){
          for (int cluster = 1; cluster < score[k].length; cluster++){
            if (score[k][cluster] < minScore) continue;
            Arrays.fill(currentGroup, 0);
            currentGroup[k] = cluster;
            currentScore = score[k][cluster];
            for (int kk = k+1; kk < k2; kk++){
              TreeMap<Integer, Integer> matches = new TreeMap<>();
              for (int j = 1; j < h-1; j++){
                for (int i = 1; i < w-1; i++){
                  int usedkk = used[kk][j][i];
                  if (usedkk != 0 && score[kk][usedkk] >= minScore &&
                      used[kk-1][j][i] == currentGroup[kk-1]){
                    Integer n = matches.get(usedkk);
                    if (n != null) matches.put(usedkk, n+1);
                    else matches.put(usedkk, 1);
                  }
                }
              }
              if (matches.size() > 0){
                int bestS = 0, bestMatch = 0;
                for (Integer i : matches.keySet()){
                  int n = matches.get(i);
                  if (n*n*8 <
                       nSquares[kk-1][currentGroup[kk-1]]*nSquares[kk][i])
                    continue;
                  if (score[kk][i] > bestS){ 
                    bestS = score[kk][i];
                    bestMatch = i;
                  }
                }
                if (bestMatch != 0){
                  currentGroup[kk] = bestMatch;
                  currentScore+= bestS;
                }else break;
              }else 
                break;
            }
            
            if (currentScore > bestScore){
              bestScore = currentScore;
              System.arraycopy(currentGroup, 0, bestGroup, 0, bestGroup.length);
            }
          }
        }
        
        // set keep to only include best group of clusters
        if (!testShapes){
          for (int k = k1; k < k2; k++)
            for (int j = 0; j < h; j++)
              for (int i = 0; i < w; i++)
                keep[k][j][i] = bestGroup[k] != 0 &&
                                used[k][j][i] == bestGroup[k];
        }
        
        // save the solution
        for (int k = k1; k < k2; k++){
          if (!testShapes)
            if (bestGroup[k] == 0) continue;
          Slice slice = scan.slices.get(k-1);
          for (int j = 1; j < h-1; j++)
            for (int i = 1; i < w-1; i++){
              String i1s = mmFormatX(i*4, slice), 
                     i2s = mmFormatX((i+1)*4, slice),
                     j1s = mmFormatY(j*4, slice),
                     j2s = mmFormatY((j+1)*4, slice);
              if (keep[k][j][i])
                solution.println(scan.id + "," + k + "," + 
                     i1s + "," + j1s + "," + i2s + "," + j1s + "," + 
                     i2s + "," + j2s + "," + i1s + "," + j2s);
              else{
                if (!testShapes){
                // extend beyond edges if needed: right
                if (keep[k][j][i-1]){
                  int ii;
                  for (ii = 0; ii < 4; ii++)
                    if (data[k][j*4    ][i*4 + ii] < lowCut ||
                        data[k][j*4 + 1][i*4 + ii] < lowCut ||
                        data[k][j*4 + 2][i*4 + ii] < lowCut ||
                        data[k][j*4 + 3][i*4 + ii] < lowCut) break;
                  if (ii > 0){
                    String iis = mmFormatX(i*4 + ii, slice);
                    solution.println(scan.id + "," + k + "," + 
                         i1s + "," + j1s + "," + iis + "," + j1s + "," + 
                         iis + "," + j2s + "," + i1s + "," + j2s);
                  }
                }
                // extend beyond edges if needed: left
                if (keep[k][j][i+1]){
                  int ii;
                  for (ii = 4; ii > 0; ii--)
                    if (data[k][j*4    ][i*4 + ii - 1] < lowCut ||
                        data[k][j*4 + 1][i*4 + ii - 1] < lowCut ||
                        data[k][j*4 + 2][i*4 + ii - 1] < lowCut ||
                        data[k][j*4 + 3][i*4 + ii - 1] < lowCut) break;
                  if (ii < 4){
                    String iis = mmFormatX(i*4 + ii, slice);
                    solution.println(scan.id + "," + k + "," + 
                         iis + "," + j1s + "," + i2s + "," + j1s + "," + 
                         i2s + "," + j2s + "," + iis + "," + j2s);
                  }
                }
                // extend beyond edges if needed: down
                if (keep[k][j-1][i]){
                  int jj;
                  for (jj = 0; jj < 4; jj++)
                    if (data[k][j*4 + jj][i*4    ] < lowCut ||
                        data[k][j*4 + jj][i*4 + 1] < lowCut ||
                        data[k][j*4 + jj][i*4 + 2] < lowCut ||
                        data[k][j*4 + jj][i*4 + 3] < lowCut) break;
                  if (jj > 0){
                    String jjs = mmFormatY(j*4 + jj, slice);
                    solution.println(scan.id + "," + k + "," + 
                         i1s + "," + j1s + "," + i2s + "," + j1s + "," + 
                         i2s + "," + jjs + "," + i1s + "," + jjs);
                  }
                }
                // extend beyond edges if needed: up
                if (keep[k][j+1][i]){
                  int jj;
                  for (jj = 4; jj > 0; jj--)
                    if (data[k][j*4 + jj - 1][i*4    ] < lowCut ||
                        data[k][j*4 + jj - 1][i*4 + 1] < lowCut ||
                        data[k][j*4 + jj - 1][i*4 + 2] < lowCut ||
                        data[k][j*4 + jj - 1][i*4 + 3] < lowCut) break;
                  if (jj < 4){
                    String jjs = mmFormatY(j*4 + jj, slice);
                    solution.println(scan.id + "," + k + "," + 
                         i1s + "," + jjs + "," + i2s + "," + jjs + "," + 
                         i2s + "," + j2s + "," + i1s + "," + j2s);
                  }
                }
                // extend beyond edges if needed: down right
                if (keep[k][j-1][i-1] && !keep[k][j][i-1] && !keep[k][j-1][i]){
                  int a = 0;
                  if (data[k][j*4][i*4] >= lowCut){ 
                    a = 1;
                    if (data[k][j*4][i*4 + 1] >= lowCut && 
                        data[k][j*4 + 1][i*4] >= lowCut){
                      a = 2;
                      if (data[k][j*4][i*4 + 2] >= lowCut && 
                          data[k][j*4 + 1][i*4 + 1] >= lowCut &&
                          data[k][j*4 + 2][i*4] >= lowCut) 
                        a = 3;
                    }
                  }
                  if (a > 0){
                    String iis = mmFormatX(i*4 + a, slice),
                           jjs = mmFormatY(j*4 + a, slice);
                    solution.println(scan.id + "," + k + "," + 
                         i1s + "," + j1s + "," +  
                         iis + "," + j1s + "," + i1s + "," + jjs);
                  }
                }
                // extend beyond edges if needed: down left
                if (keep[k][j-1][i+1] && !keep[k][j][i+1] && !keep[k][j-1][i]){
                  int a = 0;
                  if (data[k][j*4][i*4 + 3] >= lowCut){ 
                    a = 1;
                    if (data[k][j*4][i*4 + 2] >= lowCut && 
                        data[k][j*4 + 1][i*4 + 3] >= lowCut){
                      a = 2;
                      if (data[k][j*4][i*4 + 1] >= lowCut && 
                          data[k][j*4 + 1][i*4 + 2] >= lowCut &&
                          data[k][j*4 + 2][i*4 + 3] >= lowCut) 
                        a = 3;
                    }
                  }
                  if (a > 0){
                    String iis = mmFormatX(i*4 + 4 - a, slice),
                           jjs = mmFormatY(j*4 + a, slice);
                    solution.println(scan.id + "," + k + "," + 
                         iis + "," + j1s + "," +  
                         i2s + "," + j1s + "," + i2s + "," + jjs);
                  }
                }
                // extend beyond edges if needed: up right
                if (keep[k][j+1][i-1] && !keep[k][j][i-1] && !keep[k][j+1][i]){
                  int a = 0;
                  if (data[k][j*4 + 3][i*4] >= lowCut){ 
                    a = 1;
                    if (data[k][j*4 + 3][i*4 + 1] >= lowCut && 
                        data[k][j*4 + 2][i*4] >= lowCut){
                      a = 2;
                      if (data[k][j*4 + 3][i*4 + 2] >= lowCut && 
                          data[k][j*4 + 2][i*4 + 1] >= lowCut &&
                          data[k][j*4 + 1][i*4] >= lowCut) 
                        a = 3;
                    }
                  }
                  if (a > 0){
                    String iis = mmFormatX(i*4 + a, slice),
                           jjs = mmFormatY(j*4 + 4 - a, slice);
                    solution.println(scan.id + "," + k + "," + 
                         i1s + "," + jjs + "," +  
                         iis + "," + j2s + "," + i1s + "," + j2s);
                  }
                }
                // extend beyond edges if needed: up left
                if (keep[k][j+1][i+1] && !keep[k][j][i+1] && !keep[k][j+1][i]){
                  int a = 0;
                  if (data[k][j*4 + 3][i*4 + 3] >= lowCut){ 
                    a = 1;
                    if (data[k][j*4 + 3][i*4 + 2] >= lowCut && 
                        data[k][j*4 + 2][i*4 + 3] >= lowCut){
                      a = 2;
                      if (data[k][j*4 + 3][i*4 + 1] >= lowCut && 
                          data[k][j*4 + 2][i*4 + 2] >= lowCut &&
                          data[k][j*4 + 1][i*4 + 3] >= lowCut) 
                        a = 3;
                    }
                  }
                  if (a > 0){
                    String iis = mmFormatX(i*4 + 4 - a, slice),
                           jjs = mmFormatY(j*4 + 4 - a, slice);
                    solution.println(scan.id + "," + k + "," + 
                         i2s + "," + jjs + "," +  
                         i2s + "," + j2s + "," + iis + "," + j2s);
                  }
                }
                }
              }
            }
        }
      }
    }catch(Exception e){
      System.err.println(e);
    }
  }
  
  Polygon createLungPoly(int[] points, int x1, int y1, int x2, int y2){
    int dx = x2-x1, dy = y2-y1;
    P2[] p = new P2[points.length/2];
    for (int i = 0; i < p.length; i++)
      p[i] = new P2(x1 + (double)points[i*2]*dx/100, 
                    y1 + (double)points[i*2 + 1]*dy/100);
    return new Polygon(p);
  }
  
  int fillGroup(int i, int j, boolean[][] keep, int[][] used, int cluster){
    if (keep[j][i] && used[j][i] == 0){
      used[j][i] = cluster;
      if (i > 0) fillGroup(i-1, j, keep, used, cluster);
      if (i < keep[0].length-1) fillGroup(i+1, j, keep, used, cluster);
      if (j > 0) fillGroup(i, j-1, keep, used, cluster);
      if (j < keep.length-1) fillGroup(i, j+1, keep, used, cluster);
      return cluster+1;
    }else 
      return cluster;
  }
  
  String mmFormatX(double x, Slice slice){
    return String.format("%4.3f", slice.x0 + x*slice.dx);
  }
  
  String mmFormatY(double y, Slice slice){
    return String.format("%4.3f", slice.y0 + y*slice.dy);
  }
  
  short getC(DataBuffer buf, int i){
    int c = buf.getElem(i);
    if (c < 0) c+= 65536;
    if (c > Short.MAX_VALUE) c = Short.MAX_VALUE;
    return (short)c;
  }
  
  BufferedImage loadImage(String dir, String scanId, int slice){
    try{
      return ImageIO.read(new File(dir + scanId + "/pngs/" + slice + ".png"));
    }catch(Exception e){
      System.err.println(e);
      return null;
    }
  }
}

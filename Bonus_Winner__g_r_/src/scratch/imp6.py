# lung segmentation


from PIL import Image
import numpy as np
import cv2
from skimage import measure, morphology
#from tools.plot3d import plot3d
from expe.helpers.segment_lung import segment_lung_mask

from helpers.contours import read_coords, merge_contours_naive, cv2tolist

# TODO: compare with opencv
from skimage.filters import sobel

from helpers.input_data import Scan

IMAGE_SIZE = 512

def contrast(nd):
    m = nd.min()
    M = nd.max()
    return ((nd.astype(np.float32) - m) * 255 / (M-m) ).astype(np.uint8)

def display(nd):
    pilimg = Image.fromarray(contrast(nd))
    pilimg.show()

def makebox(keypoint):
    x,y = map(int,keypoint.pt)
    print(x,y)
    size = int(keypoint.size/2)
    #counter clockwise
    return [(x-size,y-size), (x-size,y+size), (x+size,y+size), (x+size,y-size)]

ID, SLICE = 'ANON_LUNG_TC227', 48  # Stuck tumor
ID, SLICE = 'ANON_LUNG_TC148', 54
ID, SLICE = 'ANON_LUNG_TC009', 46  # Try to get blob instead of box:TODO
ID, SLICE = 'ANON_LUNG_TC126', 50  # Must detect:TODO
ID, SLICE = 'ANON_LUNG_TC112', 73  # Must detect:TODO
ID, SLICE = 'ANON_LUNG_TC515', 67  # Must detect (little and stuck):TODO
ID, SLICE = 'ANON_LUNG_TC001', 69  # Try to get blob instead of box:TODO
#ID, SLICE = 'ANON_LUNG_TC002', 69  # Try to get blob instead of box:TODO

dirpath   = "/home/gerey/hms_lung/data/example_extracted/%s/" % (ID, )

# In valid directory

#ID, SLICE = 'ANON_LUNG_TC604', 113  # Try to get blob instead of box:TODO

#imgname0   = "/home/gerey/hms_lung/data/example_extracted_valid/%s/pngs/%s.png" % (ID, SLICE)
#imgname1   = "/home/gerey/hms_lung/data/example_extracted_valid/%s/pngs/%s.png" % (ID, SLICE+1)


scan = Scan(dirpath)

scan3D = scan.scan3D()

data0 = scan3D[SLICE]
#data1 = cv2.imread(imgname1, cv2.IMREAD_UNCHANGED)

#data1 = data1.astype(np.float32)

data0 = data0.astype(np.float32)

display(data0)

segmented = segment_lung_mask(scan3D, False)
#plot3d(segmented)
display(segmented[SLICE-1])

#segmented = segment_lung_mask(data0, True)

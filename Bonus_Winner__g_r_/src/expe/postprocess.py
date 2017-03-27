# opening/closing of whole 3D answers.
from itertools import tee

import numpy as np
import cv2
import skimage
from scipy import ndimage

from compute_scores import solution2dict
from helpers.contours import convert_coords, pixelToMM, cv2tolist
from helpers.input_data import Dataset, grouper
from helpers.misc import display, contrast, flatten
from imprediction3 import findContours

IMAGE_SIZE = 512

def make_sol3d(sol, scan):
    res = np.zeros([scan.nb_slices(), IMAGE_SIZE, IMAGE_SIZE])
    for slice_id, polygons in sol[scan.id()].items():
        auxpath       = scan.aux[slice_id]
        for polygon in polygons:
            points = np.array(convert_coords(polygon, auxpath), np.int32).reshape((-1, 1, 2))
            #res[slice_id-1] = cv2.fillPoly(res[slice_id-1], [points], True)
            cv2.drawContours(res[slice_id-1], [points], 0, color=1, thickness=-1)
            #display(contrast(res[slice_id-1]))
    return res.astype(bool)


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def enhance_by_slice(gen):
    """
    for (m1,s1),(m2,s2) in pairwise(gen):
        if np.any(m1) or np.any(m2):
            newmask = np.logical_or(m1, ndimage.binary_dilation(m2, iterations=2))
            mean = s2[newmask].mean()
            newmask[abs(s2-mean)>42] = 0
            m2[newmask] = 1
    """
    for (m1,s1),(m2,s2) in pairwise(gen):
        if 1:
            if np.any(m2):
                mu = s2[m2].mean()
                newmask = np.logical_and(skimage.morphology.erosion(m2), abs(s1-mu)<42)
                m1[newmask] = 1


def morph_open(mask3D):
   return skimage.morphology.binary_opening(mask3D, skimage.morphology.ball(2))

def morph_close(mask3D):
    return skimage.morphology.binary_closing(mask3D, skimage.morphology.ball(2))

def enhance_scan(mask3D, scan3D):

    for _ in range(1):
        for axis in range(3):
            enhance_by_slice(zip(np.rollaxis(mask3D,axis),
                                 np.rollaxis(scan3D,axis)))
            enhance_by_slice(zip(np.rollaxis(mask3D,axis)[::-1],
                                 np.rollaxis(scan3D,axis)[::-1]))
            mask3D[...] = morph_close(mask3D)

    mask3D[...] = skimage.morphology.binary_opening(mask3D,np.ones((2,1,1)))


def enhance(sol, dataset):

   for scan in dataset.scans():
        print(scan.id())
        scan3D = scan.scan3D()
        sol3D = make_sol3d(sol, scan)
        enhance_scan(sol3D, scan3D)
        for idx, slice in enumerate(sol3D):
            cnts = findContours(slice)
            for cnt in cnts:
                coords = flatten(pixelToMM(scan.aux[idx+1],x,y) for (x,y) in cv2tolist(cnt))
                yield [scan.id(), idx+1] + ["%.4f" % xy for xy in coords]


if __name__ == '__main__':

    dataset = Dataset("/home/gerey/hms_lung/data/extract", withGT=False)
    csvpath = "/home/gerey/hms_lung/single.csv"
    output  = "/home/gerey/hms_lung/single_enhanced2.csv"
    # dataset = Dataset("/home/gerey/hms_lung/data/provisional_extracted_no_gt", withGT=False)
    # csvpath = "/home/gerey/hms_lung/predictions27.csv"
    # output  = "/home/gerey/hms_lung/enhanced27.csv"
    sol = solution2dict(csvpath)

    with open(output, "w") as f:
        for s in enhance(sol,dataset):
            f.write(",".join(map(str,s)))
            f.write("\n")


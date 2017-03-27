# Get blobs from original image (distance map + watershed)
from scipy import ndimage

import numpy as np
import cv2
# TODO try with opencv
from skimage.feature import peak_local_max
from skimage.morphology import watershed

from helpers.misc import contrast


def dist_watershed(data):

    data = contrast(data)

    _, bin = cv2.threshold(data,0,255,cv2.THRESH_OTSU)

    #remove white dots. ok on ANON_LUNG_TC148 #54
    #kernel = np.ones((7,7),np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    #nowhitedots = cv2.erode(bin, kernel, iterations = 3)
    nowhitedots = cv2.morphologyEx(bin,cv2.MORPH_OPEN, kernel, iterations = 1)

    dist = ndimage.distance_transform_edt(nowhitedots)
    localMax = peak_local_max(dist, indices=False, min_distance=10,
                              labels=nowhitedots)

    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-dist, markers, mask=nowhitedots)

    res = []
    for label in np.unique(labels):
        # skip background
        if label == 0:
            continue

        mask = np.zeros(data.shape, dtype="uint8")
        mask[labels == label] = 255

        _, cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        res += cnts

    return res

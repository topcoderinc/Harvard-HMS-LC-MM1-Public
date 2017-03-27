# Copy/Pasted from imprediction, without 'manual' filtering
# TODO: factorize

import cv2
import numpy as np
from helpers.misc import makebox, display, contrast, flatten

from helpers.contours import pixelToMM, merge_contours_naive, cv2tolist

def contouring_binsym(data):

    data = contrast(data)
    #display(data)

    _, bin = cv2.threshold(data,0,255,cv2.THRESH_OTSU)

    # keep only what doesn't appear on the other side
    sym = bin[:,::-1]
    diff = bin-sym
    _, diff = cv2.threshold(diff,128,255,cv2.THRESH_BINARY)   # clean the '1' which mess up with findcontours

    #remove white dots. ok on ANON_LUNG_TC148 #54
    kernel = np.ones((3,3),np.uint8)
    nowhitedots = cv2.morphologyEx(diff,cv2.MORPH_OPEN, kernel, iterations = 2)

    #fill black holes
    kernel = np.ones((3,3),np.uint8)
    noblackholes = cv2.morphologyEx(nowhitedots,cv2.MORPH_CLOSE, kernel, iterations = 2)

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0;
    params.maxThreshold = 255;
    params.thresholdStep = 55;

    # Filter by Color
    params.filterByColor = False
    params.blobColor = 255   # detect bright spot

    # Filter by Area.
    params.filterByArea = False
    params.minArea = 1500

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = False
    #params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.1 #More important than circularity actually

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(noblackholes)

    # We'll keep the CC corresponding to these keypoints.
    # Naive version :
    #  get contours of all CC, and test if one of a point is in contour

    # NB: COMP allows to get external boundaries, we don't care about holes
    _, contours, _ = cv2.findContours(noblackholes,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    kept=[]
    for cnt in contours:
        for kp in keypoints:
            if cv2.pointPolygonTest(cnt, kp.pt, False) >= 0:
                kept.append(cnt)
                break

    return kept


def contouring_binsym_all(data):

    data = contrast(data)
    #display(data)

    _, bin = cv2.threshold(data,0,255,cv2.THRESH_OTSU)

    # keep only what doesn't appear on the other side
    sym = bin[:,::-1]
    diff = bin-sym
    _, diff = cv2.threshold(diff,128,255,cv2.THRESH_BINARY)   # clean the '1' which mess up with findcontours

    #remove white dots. ok on ANON_LUNG_TC148 #54
    kernel = np.ones((3,3),np.uint8)
    nowhitedots = cv2.morphologyEx(diff,cv2.MORPH_OPEN, kernel, iterations = 2)

    #fill black holes
    kernel = np.ones((3,3),np.uint8)
    noblackholes = cv2.morphologyEx(nowhitedots,cv2.MORPH_CLOSE, kernel, iterations = 2)

    # NB: COMP allows to get external boundaries, we don't care about holes
    _, contours, _ = cv2.findContours(noblackholes,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    return contours

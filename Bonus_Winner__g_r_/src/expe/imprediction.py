# Copyright 2017 . All Rights Reserved.

# Contouring by basic image processing operation
# Version1 : simply detect dyssimetric blobs.

import numpy as np
import cv2
#from helpers.contours import read_coords
from helpers.input_data import Dataset
from helpers.contours import pixelToMM, merge_contours_naive, cv2tolist
from helpers.misc import makebox, display, contrast, flatten
import os

lungleft=150
lungtop=230
lungright=350
lungbot=374


def validate(keypoint):
    x,y = map(int,keypoint.pt)
    nottoofar = (lungleft < x < lungright) and \
                (lungtop  < y < lungbot)
    nottoocentered = abs(x-256) > 26
    return nottoofar and nottoocentered


def contouring(data):

    data = data.astype(np.float32)

    sym = data[:,::-1]
    symd = (sym-data)**2
    symd = contrast((symd-symd.min())*(data-data.min()))
    #display(symd)

    # TODO: more robust opening!
    kernel = np.ones((6,6),np.uint8)
    symd = cv2.morphologyEx(symd,cv2.MORPH_OPEN, kernel, iterations = 2)
    symd = contrast(symd)
    #display(symd)

    # Set filters (disable them, mostly)
    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 100
    params.maxThreshold = 200
    params.thresholdStep = 50

    params.filterByColor = False
    params.blobColor = 255   # detect bright spot

    params.filterByArea = False
    #params.minArea = 1500

    params.filterByCircularity = True
    params.minCircularity = 0.3

    params.filterByConvexity = False
    #params.minConvexity = 0.87

    params.filterByInertia = False
    #params.minInertiaRatio = 0.01

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(symd)
    boxes = [makebox(kp) for kp in keypoints if validate(kp)]

    return merge_contours_naive(boxes)


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

    # Eliminate candidates too far or too close, etc.
    keypoints = list(filter(validate, keypoints))  # arggllllll si non list, iteration epuisé en une fois!!

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

    # Now get convex hull (more likely)
    return [ cv2tolist(cnt) for cnt in kept ]
    return [ cv2tolist(cv2.convexHull(cnt, clockwise=False)) for cnt in kept ]


if __name__ ==  '__main__':

    #dataset = Dataset("/home/gerey/hms_lung/data/provisional_extracted_no_gt", withGT=False)
    dataset = Dataset("/home/gerey/hms_lung/data/example_extracted_valid", withGT=False)
    #dataset = Dataset("/home/gerey/hms_lung/data/extract", withGT=False)

    scan_index = 1
    solutions = []
    for scan in dataset.scans():
        for id, img, aux in scan.images_aux():
            scan_id = scan.id()
            # only process middle slices
            # TODO: expend once false negatives are cleared
            if scan.nb_slices()/4 < id < scan.nb_slices()*1.7/3:
               print(scan_id, id, len(slices))
               contours = contouring_binsym(img)
               for contour in contours:
                   coords = flatten(pixelToMM(aux[id],x,y) for (x,y) in contour)
                   solutions.append([scan_id, id] + ["%.4f" % xy for xy in coords])

    with open("/home/gerey/hms_lung/predictions_valid8.csv","w") as f:
    # with open("/home/gerey/hms_lung/example_predictions8.csv","w") as f:
        for s in solutions:
            f.write(",".join(map(str,s)))
            f.write("\n")

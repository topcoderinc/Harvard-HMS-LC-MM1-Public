# Copyright 2017 . All Rights Reserved.

# Contouring by basic image processing operation
# Version1 : simply detect dyssimetric blobs.
import multiprocessing
from itertools import islice

import numpy as np
import cv2

from sklearn.externals import joblib

#from helpers.contours import read_coords
from helpers.expand_blob import expand_blob
from helpers.input_data import Dataset
from helpers.contours import pixelToMM, merge_contours_naive, cv2tolist
from helpers.misc import makebox, display, contrast, flatten

from extract_features import contour_extractors
from extract_features import features_from_contour
from expe.helpers.segment_lung import segment_lung_mask

IMAGE_SIZE = 512

precision_model = "/home/gerey/hms_lung/data/example_extracted/precision_randomforest6-%s.clf"
recall_model = "/home/gerey/hms_lung/data/example_extracted/recall_randomforest6-%s.clf"

precision_clf = [joblib.load(precision_model % i) for i in range(len(contour_extractors))]
recall_clf =    [joblib.load(recall_model    % i) for i in range(len(contour_extractors))]

USE_LUNG = False
MAX_MASK = True

def merge_and_filter(cnts, slice, scan3D, lung):
    """ naive version """
    win = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)  # bool not supported by cv2

    # Merge #####
    for cnt in cnts:
        points = np.array(cnt, np.int32).reshape((-1, 1, 2))
        win = cv2.fillPoly(win, [points], 1)

    if lung is None:
        return win

    if MAX_MASK:
        mask = lung
    else:
        # Intersect with lung's convex hull ####
        _, lungcnts, _ = cv2.findContours(lung[slice-1].astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)  # bool not supported by cv2
        for cnt in lungcnts:
            lungcvx = cv2.convexHull(cnt)
            points = np.array(lungcvx, np.int32).reshape((-1, 1, 2))
            mask = cv2.fillPoly(mask, [points], 1)

    res = win & mask
    return res if res.any() else win


def findContours(win):
    _, contours, _ = cv2.findContours(win.astype(np.uint8) ,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours


def collect_contours(img, slice, nbslices):
    cnts = []
    for idx, contour_extractor in enumerate(contour_extractors):
        for cnt in contour_extractor(img):
            if (len(cnt)>=5):  # needed for fitEllipse
                features = [slice, slice/nbslices] + features_from_contour(cnt, img)
                features = [features] #single sample
                precision = precision_clf[idx].predict(features)[0]
                if 1:
                   recall = recall_clf[idx].predict(features)[0]
                   print(idx, precision ,recall)
                thresholds = [0.33, 0.42, 0.6]
                if precision > thresholds[idx]:
                    cnts.append(cnt)
    return cnts


def compute(params):
    scan_id, slice, nbslices, aux, scan3D, lung = params
    img = scan3D[slice-1]
    print(scan_id, slice)
    cnts = collect_contours(img, slice, nbslices)

    res = []
    if cnts:
        # Keep best candidate
        merged = merge_and_filter(cnts, slice, scan3D, lung)
        corrected_cnts = findContours(merged)
        for cnt in corrected_cnts:
            # cnt = expand_blob(scan3D[slice-1], cnt)
            coords = flatten(pixelToMM(aux,x,y) for (x,y) in cv2tolist(cnt))
            res.append([scan_id, slice] + ["%.4f" % xy for xy in coords])
    return res

def gen_params():
    for scan in dataset.scans():
        scan3D = scan.scan3D()
        if USE_LUNG:
            lung = segment_lung_mask(scan3D, fill_lung_structures=True)
            if MAX_MASK:
                lung = np.any(lung, axis=0)
        else:
            lung = None
        for scan_id, slice_idx, aux in scan.gen_aux():
            yield scan_id, slice_idx, scan.nb_slices(), aux, scan3D, lung


if __name__ ==  '__main__':

    # dataset = Dataset("/home/gerey/hms_lung/data/provisional_extracted_no_gt", withGT=False)
    dataset = Dataset("/home/gerey/hms_lung/data/example_extracted_valid_small2", withGT=False)

    pool = multiprocessing.Pool(30)
    solutions = flatten(pool.imap( compute, gen_params() ))
    # solutions = flatten(map( compute, gen_params() ))

    with open("/home/gerey/hms_lung/exemple_predictions_sample2_33.csv","w") as f:
    # with open("/home/gerey/hms_lung/predictions31.csv","w") as f:
        for s in solutions:
            f.write(",".join(map(str,s)))
            f.write("\n")

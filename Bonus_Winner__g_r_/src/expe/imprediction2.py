# Copyright 2017 . All Rights Reserved.

# Contouring by basic image processing operation
# Version1 : simply detect dyssimetric blobs.
import multiprocessing

from sklearn.externals import joblib

#from helpers.contours import read_coords
from helpers.input_data import Dataset
from helpers.contours import pixelToMM, merge_contours_naive, cv2tolist
from helpers.misc import makebox, display, contrast, flatten

from extract_features import contour_extractors
from extract_features import features_from_contour


if __name__ ==  '__main__':

    # dataset = Dataset("/home/gerey/hms_lung/data/provisional_extracted_no_gt", withGT=False)
    #dataset = Dataset("/home/gerey/hms_lung/data/example_extracted_valid", withGT=False)
    dataset = Dataset("/home/gerey/hms_lung/data/example_extracted_valid", withGT=False)
    precision_model = "/home/gerey/hms_lung/data/example_extracted/precision_randomforest4.clf"
    recall_model = "/home/gerey/hms_lung/data/example_extracted/recall_randomforest4.clf"

    precision_clf = joblib.load(precision_model)
    recall_clf = joblib.load(recall_model)

    def compute(params):
        scan_id, slice, nbslices, img, aux = params
        print(scan_id, slice)
        res = []
        for idx, contour_extractor in enumerate(contour_extractors):
            for cnt in contour_extractor(img):
                if (len(cnt)>=5):  # needed for fitEllipse
                    features = [idx, slice, slice/nbslices] + features_from_contour(cnt, img)
                    features = [features] #single sample
                    precision = precision_clf.predict(features)
                    if precision > 0.01:
                       recall = recall_clf.predict(features)
                       if recall > 0.01:
                           coords = flatten(pixelToMM(aux[slice],x,y) for (x,y) in cv2tolist(cnt))
                           res.append([scan_id, slice] + ["%.4f" % xy for xy in coords])
        return res

    pool = multiprocessing.Pool(40)
    solutions = flatten(pool.imap( compute, dataset.images_and_aux() ))

    with open("/home/gerey/hms_lung/example_predictions_valid12.csv","w") as f:
    # with open("/home/gerey/hms_lung/example_predictions8.csv","w") as f:
        for s in solutions:
            f.write(",".join(map(str,s)))
            f.write("\n")

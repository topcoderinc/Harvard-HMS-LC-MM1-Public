# Make a dataset with extracted features and target
import multiprocessing
import numpy as np
import cv2

import itertools

from features_extractor.bygrayrange import by_gray_range
from features_extractor.asymetric_blobs import contouring_binsym_all
from features_extractor.dist_watershed  import dist_watershed

from features_extractor.features import features_from_contour

from helpers.input_data import Dataset

from helpers.misc import flatten

# use all blob extractions at disposal
contour_extractors = [contouring_binsym_all, dist_watershed, by_gray_range]


def compute_features(params):

    id, slice, nbslices, image, target = params

    truth_area = target.sum()
    res = []

    for idx, contour_extractor in enumerate(contour_extractors):

        for cnt in contour_extractor(image):

            # filter too small blob, since some extractor (fitEllipse) needs at least 5 points.
            if (len(cnt)>=5):
                # Measure area of intersection/union as regression target (1 = perfect match)
                imgmask = np.zeros(image.shape[:2]).astype('uint8')
                cv2.drawContours(imgmask, [cnt], 0, color=255, thickness=-1)
                mask = (imgmask == 255)
                blob_area = mask.sum()
                tp_area = target[mask].sum()
                precision = tp_area / blob_area
                recall    = tp_area / truth_area  if truth_area else 0

                res.append([id, precision, recall, idx, slice, slice/nbslices] + features_from_contour(cnt, image))
    return res


def generate_features(iter):

    pool = multiprocessing.Pool(10)
    return flatten(pool.imap( compute_features, iter ))


def main(dataset, output):

    # it = iter(generate_features(dataset))
    # with open(output, "w") as f:
    #     for _ in range(2):
    #         print(" ".join(map(str, next(it))), file=f)
    with open(output, "w") as f:
        for features in generate_features(dataset):
             print(" ".join(map(str, features)), file=f)


if __name__ ==  '__main__':

    output = "/home/gerey/hms_lung/data/example_extracted/features6.ssv"

    train = Dataset("/home/gerey/hms_lung/data/example_extracted")
    train_iter = train.images_and_targets()

    if 1:
        valid = Dataset("/home/gerey/hms_lung/data/example_extracted_valid")
        valid_iter = valid.images_and_targets()
        full_iter = itertools.chain(train_iter, valid_iter)
    else:
        full_iter = train_iter

    main(full_iter, output)


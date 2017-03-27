# extract features from contour

import cv2
import numpy as np
from sklearn.metrics.cluster import entropy

def features_from_contour(cnt, img):

    # moments
    m = cv2.moments(cnt)
    # centroid
    cx, cy = m['m10']/m['m00'], m['m01']/m['m00']
    area = m ['m00']

    perimeter = cv2.arcLength(cnt, True)

    # bounding box
    bbx, bby, bbw, bbh = cv2.boundingRect(cnt)

    #  diameter of circle with same area as region
    diameter = np.sqrt(4 * area / np.pi)
    #  ratio of area of region to area of bounding box
    extent = area / (bbw * bbh)

    convexHull = cv2.convexHull(cnt)
    convexArea = cv2.contourArea(convexHull)
    solidity = area / convexArea

    # best-fitting ellipse.
    centre, axes, angle = cv2.fitEllipse(cnt)
    minor, major = sorted(axes)
    eccentricity = np.sqrt(1 - (minor / major) ** 2)
    orientation = angle
    ecx, ecy = centre  # x,y

    # Texture features
    imgmask = np.zeros(img.shape[:2]).astype('uint8')
    cv2.drawContours(imgmask, [cnt], 0, color=255, thickness=-1)
    mask = (imgmask==255)
    values = img[mask]

    minIntensity = values.min()
    maxIntensity = values.max()
    meanIntensity = np.mean(values)

    entr = entropy(values)

    hist, _ = np.histogram(values, density=True)

    # TODO:
    # http://scikit-image.org/docs/dev/api/skimage.feature.html
    # SURF, SIFT

    minGlobal = img.min()
    maxGlobal = img.max()
    meanGlobal = np.mean(img)

    _, sortedmoments = zip(*sorted(m.items()))
    hu = cv2.HuMoments(m)

    #TODO: inertia, circularity, et plus si manque d'affinité.
#
    return [cx, cy, area, perimeter, bbx, bby, bbw, bbh, diameter, extent, solidity,
            axes[0], minor, major, minor/(major+0.01), angle, eccentricity, orientation, ecx, ecy,
            minIntensity, maxIntensity, meanIntensity,
            minGlobal, maxGlobal, meanGlobal, entr, ] \
           + list(hist) + list(sortedmoments) + list(hu.flatten())


# Crude blob detection by grey level #########  Cf imp7 for visu
import cv2
import numpy as np

def by_gray_range(data):

    data0 = np.copy(data)
    data0[data0<1020] = 0
    data0[data0>1120] = 0

    kernel =cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
    data0 = cv2.morphologyEx(data0,cv2.MORPH_OPEN, kernel, iterations = 1)

    kernel =cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
    data0 = cv2.morphologyEx(data0,cv2.MORPH_CLOSE, kernel, iterations = 1)

    data0 = (data0 > 100).astype(np.uint8)

    _, cnts, _ = cv2.findContours(data0, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    return cnts




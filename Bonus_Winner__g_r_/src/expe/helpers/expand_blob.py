import cv2
import numpy as np

IMAGE_SIZE = 512

def expand_blob(img, cnt):

    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    points = np.array(cnt, np.int32).reshape((-1, 1, 2))
    mask = cv2.fillPoly(mask, [points], 1)
    mask = mask.astype(bool)  # necessary for proper masking !

    process=True
    it=0
    while process and it<2:
        # remove points too far from mean
        #print(img[mask])
        mean =    np.mean(img[mask])
        stddev =  np.std(img[mask])
        mask[np.abs(img - mean) > 4*stddev] = 0
        kernel =cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        mask = cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_CLOSE, kernel, iterations = 1)
        mask = mask.astype(bool)

        # add adjacent points
        mean =    np.mean(img[mask])
        stddev =  np.std(img[mask])
        kernel =cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations = 2)
        delta = np.logical_and( dilated, np.logical_not(mask) )
        newpix = np.logical_and( delta, abs(img - mean) < stddev*0.7 )
        mask = np.logical_or(mask, newpix)
        #print(np.sum(newpix),np.sum(delta))
        mask = cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_OPEN, kernel, iterations = 1)
        mask = mask.astype(bool)
        process = np.sum(newpix)*3 > np.sum(delta)
        it+=1

    _, newcnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    return newcnt[0] if newcnt else cnt


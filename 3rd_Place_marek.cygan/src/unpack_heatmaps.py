import os
import glob
import cPickle as pickle
import cv2
import numpy as np

for s in open('scans_all.csv', 'r').readlines():
    v = s.split(',')
    patient = v[0]
    sid = v[1]
    jpgs = pickle.load(open('heatmaps/'+patient+'_'+sid+'.p'))
    img = cv2.imdecode(np.array(bytearray(jpgs[0])), -1)
    cv2.imwrite('unpacked_heatmaps/'+patient+'_'+sid+'_0.jpg', img)

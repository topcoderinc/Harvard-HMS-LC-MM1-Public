import os
import glob
import cPickle as pickle
import cv2
import numpy as np

dict = {}

for s in open('scans_all.csv', 'r').readlines():
    v = s.split(',')
    patient = v[0]
    sid = v[1]
    d = float(v[6])
    if patient not in dict:
        print patient, d
        dict[patient] = 0
    jpgs = pickle.load(open('heatmaps512/'+patient+'_'+sid+'.p'))
    img = cv2.imdecode(np.array(bytearray(jpgs[0])), -1)
    dict[patient] += np.sum(img) / 255.0 * d
    cv2.imwrite('unpacked_heatmaps/'+patient+'_'+sid+'_0.jpg', img)

f = open('weights.csv', 'w')
for s in open('scans_all.csv', 'r').readlines():
    v = s.split(',')
    patient = v[0]
    sid = v[1]
    print >>f, '{},{},{}'.format(patient,sid,dict[patient])
f.close()

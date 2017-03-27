import numpy as np
import os.path
import scipy
import argparse
import math
from sklearn.preprocessing import normalize

caffe_root = '/home/lungcancer/sandbox/lung_cancer/caffe-segnet/' 			# Change this to the absolute directoy to SegNet Caffe
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--root_path', type=str, required=True)
parser.add_argument('--text_file_name', type=str, required=True)
args = parser.parse_args()

caffe.set_mode_gpu()

net = caffe.Net(args.model, args.weights, caffe.TEST)


root_path = args.root_path
text_file_name = args.text_file_name


with open( text_file_name, 'r') as text_file:
    path_to_images = text_file.readlines()


print 'Running inference (please wait) ...'
for path_to_image in path_to_images:

    path_to_image = path_to_image.split('/')
    slice_id = path_to_image[-1].rstrip('\n')
    scan_id = path_to_image[-3]

    result_directory = root_path + scan_id + '/results/'
    if not os.path.exists( result_directory ):
        os.makedirs( result_directory )

    net.forward()

    #image = net.blobs['data'].data
    #    label = net.blobs['data'].name
    predicted = net.blobs['prob'].data
    #image = np.squeeze(image[0,:,:,:])
    output = np.squeeze(predicted[0,:,:,:])
    ind = np.argmax(output, axis=0)

    r = ind.copy()
    g = ind.copy()
    b = ind.copy()
    #	r_gt = label.copy()
    #	g_gt = label.copy()
    #	b_gt = label.copy()

    not_tumor = [0,0,0]
    tumor = [255,255,255]

    label_colours = np.array([not_tumor, tumor])
    for l in range(0, 2):
        r[ind==l] = label_colours[l,0]
        g[ind==l] = label_colours[l,1]
        b[ind==l] = label_colours[l,2]
    #		r_gt[label==l] = label_colours[l,0]
    #		g_gt[label==l] = label_colours[l,1]
    #		b_gt[label==l] = label_colours[l,2]

    rgb = np.zeros((ind.shape[0], ind.shape[1], 3))
    rgb[:,:,0] = r/255.0
    rgb[:,:,1] = g/255.0
    rgb[:,:,2] = b/255.0
    #    rgb_gt = np.zeros((ind.shape[0], ind.shape[1], 3))
    #    rgb_gt[:,:,0] = r_gt/255.0
    #    rgb_gt[:,:,1] = g_gt/255.0
    #    rgb_gt[:,:,2] = b_gt/255.0

    #image = image/255.0

    #image = np.transpose(image, (1,2,0))
    #output = np.transpose(output, (1,2,0))
    #image = image[:,:,(2,1,0)]

    path_to_image = result_directory + slice_id

    max_value = np.amax( rgb )
    # Rescale image intensity
    if( max_value != 0 ):
        rgb = (rgb / max_value) * 255
    result = rgb.astype( np.uint8 )

    scipy.misc.toimage(result, cmin=0.0, cmax=255).save( path_to_image )

print 'Success!'


#!/usr/bin/env python
from hms_lung_tumor import HMSLungTumor

if __name__ == "__main__":
    gt = HMSLungTumor()

    # Path to patient files
    print 'Generating caffe traning files...'

    root_path = '/home/thomio/datasets/lung_tumor/example_extracted'
    gt.compute_images_mean( root_path )

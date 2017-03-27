#!/usr/bin/env python
from hms_lung_tumor import HMSLungTumor

if __name__ == "__main__":
    gt = HMSLungTumor()

    # Path to patient files
    root_path = '/home/thomio/datasets/lung_tumor/example_extracted/'

    print 'Generating caffe traning files...'

    gt.save_file_with_imgs_paths( root_path, 'train' )
    gt.save_file_with_imgs_paths( root_path, 'test' )

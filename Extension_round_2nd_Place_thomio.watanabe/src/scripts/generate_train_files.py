#!/usr/bin/env python
from libs.pre_processing import *

if __name__ == "__main__":
    # Path to scans
    root_path = '/home/thomio/datasets/lung_tumor/example_extracted/'

    print 'Generating caffe traning files...'
    save_file_with_imgs_paths( root_path, 'train' )
    save_file_with_imgs_paths( root_path, 'test' )

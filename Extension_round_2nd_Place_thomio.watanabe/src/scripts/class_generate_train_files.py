#!/usr/bin/env python
from libs.image_classification import *

# Generate GoogLeNet train files

if __name__ == "__main__":
    # Path to patient files
    root_path = '/home/thomio/datasets/lung_tumor/example_extracted/'

    print 'Generating caffe traning files...'
    save_imgs_paths( root_path, 'train' )
    save_imgs_paths( root_path, 'test' )

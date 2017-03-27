#!/usr/bin/env python
from libs.pre_processing import *

if __name__ == "__main__":
    # Path to patient files
    root_path = '/home/thomio/datasets/lung_tumor/example_extracted/'

    # Separate the dataset in train and test directories before runing this
    print 'Rename images...'
    rename_images( root_path, 'test' )
    rename_images( root_path, 'train' )

#!/usr/bin/env python
from libs.pre_processing import *

if __name__ == "__main__":
    # Path to patient files
    root_path = '/home/thomio/datasets/lung_tumor/example_extracted/'

    # Separate the dataset in train and test directories before runing this
    print 'Concat images...'
    concat_images( root_path, 'test' )
    save_path_to_concat( root_path, 'test' )

    concat_images( root_path, 'train' )
    save_path_to_concat( root_path, 'train' )

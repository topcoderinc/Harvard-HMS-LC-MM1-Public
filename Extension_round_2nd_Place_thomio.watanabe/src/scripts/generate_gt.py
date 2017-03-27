#!/usr/bin/env python
from libs.hms_lung_tumor import *

if __name__ == "__main__":
    # Path to patient files
    root_path = '/home/thomio/datasets/lung_tumor/example_extracted/'

    # Separate the dataset in train and test directories before runing this
    print 'Generating ground truth...'
    generate_gt( root_path, 'test' )
    generate_gt( root_path, 'train' )


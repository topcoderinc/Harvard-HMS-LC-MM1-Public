#!/usr/bin/env python
from hms_lung_tumor import HMSLungTumor

if __name__ == "__main__":
    gt = HMSLungTumor()

    # Path to patient files
    root_path = '/home/thomio/datasets/lung_tumor/example_extracted/'

    # Separate the dataset in train and test directories before runing this
    print 'Generating ground truth...'
    gt.generate_gt( root_path, 'train' )
    gt.generate_gt( root_path, 'test' )

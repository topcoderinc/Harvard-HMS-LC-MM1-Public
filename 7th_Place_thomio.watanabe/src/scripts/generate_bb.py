#!/usr/bin/env python
from hms_lung_tumor import HMSLungTumor

if __name__ == "__main__":
    gt = HMSLungTumor()

    print 'Generating bouding boxes...'

    # Path to patient scans
    root_path = '/home/thomio/datasets/lung_tumor/provisional_extracted_no_gt/'
    gt.generate_bb( root_path, 'results.csv' )

#    root_path = '/home/thomio/datasets/lung_tumor/example_extracted/test/'
#    gt.generate_bb( root_path, 'test.csv' )

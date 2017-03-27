#!/usr/bin/env python
from hms_lung_tumor import HMSLungTumor

if __name__ == "__main__":
    gt = HMSLungTumor()

    # Path to patient files
    root_path = '/home/thomio/datasets/lung_tumor/provisional_extracted_no_gt/'

    print 'Converting provisional images...'
    gt.convert_provisional_images( root_path )

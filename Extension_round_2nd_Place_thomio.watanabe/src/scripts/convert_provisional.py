#!/usr/bin/env python
from libs.pre_processing import *

if __name__ == "__main__":

    # Path to inference scans
    root_path = '/home/thomio/datasets/lung_tumor/provisional_extracted_no_gt/'

    print 'Converting provisional images...'
    convert_provisional_images( root_path )

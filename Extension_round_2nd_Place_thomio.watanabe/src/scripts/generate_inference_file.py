#!/usr/bin/env python
from libs.pre_processing import *

if __name__ == "__main__":

    # Path to inference scans
    root_path = '/home/thomio/datasets/lung_tumor/provisional_extracted_no_gt/'

    print 'Generating caffe inference file...'
    save_path_to_inference( root_path )

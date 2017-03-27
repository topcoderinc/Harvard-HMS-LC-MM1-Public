#!/usr/bin/env python
from libs.hms_lung_tumor import *

if __name__ == "__main__":

    print 'Generating bouding boxes...'

    # Path to patient scans
#    root_path = '/home/thomio/datasets/lung_tumor/provisional_extracted_no_gt/'
#    generate_bb( root_path, 'results.csv' )

#    path_to_file = '/home/thomio/sandbox/lung_cancer/caffe/classification/inference_slices.txt'
#    path_to_file = '/home/thomio/sandbox/lung_cancer/caffe/classification/inference_slices_long_train.txt'
#    generate_bb_from_file( path_to_file, 'results.csv' )


    root_path = '/home/thomio/datasets/lung_tumor/example_extracted/test/'
    generate_bb( root_path, 'test.csv' )
#    path_to_file = '/home/thomio/sandbox/lung_cancer/caffe/dataset/test_tumors_only.txt'
#    generate_bb_from_file( path_to_file, 'test_only_tumor.csv' )

#    path_to_file = '/home/thomio/sandbox/lung_cancer/caffe/classification/test_slices.txt'
#    generate_bb_from_file( path_to_file, 'test_only_tumor_class_01.csv' )
#    path_to_file = '/home/thomio/sandbox/lung_cancer/caffe/classification/test_slices_long_train.txt'
#    generate_bb_from_file( path_to_file, 'test_only_tumor_class_02.csv' )

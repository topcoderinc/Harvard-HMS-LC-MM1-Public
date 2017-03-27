#!/usr/bin/env python
from hms_lung_tumor import HMSLungTumor

if __name__ == "__main__":
    gt = HMSLungTumor()

    print 'Calculathing classes weights...'

    root_path = '/home/thomio/sandbox/lung_cancer/caffe/dataset/train_lung_gt_mix.txt'
    gt.calculate_class_weighting( root_path )

#    root_path = '/home/thomio/sandbox/lung_cancer/caffe/dataset/all_images.txt'
#    gt.calculate_class_weighting( root_path )

    # Path to patient scans
#    root_path = '/home/thomio/datasets/lung_tumor/example_extracted/train/'
#    gt.calculate_class_weighting_using_dir( root_path )

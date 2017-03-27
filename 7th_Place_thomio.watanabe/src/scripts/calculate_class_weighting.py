#!/usr/bin/env python
from hms_lung_tumor import HMSLungTumor

if __name__ == "__main__":
    gt = HMSLungTumor()

    print 'Calculathing classes weights...'

    # Path to patient scans
    root_path = '/home/thomio/sandbox/lung_cancer/caffe/dataset/train_mix_02.txt'
    gt.calculate_class_weighting( root_path )

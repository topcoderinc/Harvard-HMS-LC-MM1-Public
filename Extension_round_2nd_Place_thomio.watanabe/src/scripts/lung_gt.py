#!/usr/bin/env python
from hms_lung_tumor import HMSLungTumor

if __name__ == "__main__":
    gt = HMSLungTumor()

    root_path = '/home/thomio/datasets/lung_tumor/example_extracted/test/'
    gt.lung_gt( root_path )
    gt.save_lung_gt_paths( root_path, 'test' )

    root_path = '/home/thomio/datasets/lung_tumor/example_extracted/train/'
    gt.lung_gt( root_path )
    gt.save_lung_gt_paths( root_path, 'train' )

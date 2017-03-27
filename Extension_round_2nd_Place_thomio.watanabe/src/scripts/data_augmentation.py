#!/usr/bin/env python
from libs.data_augmentation import *

if __name__ == "__main__":
    # Path to patient files
    root_path = '/home/thomio/datasets/lung_tumor/example_extracted/'

    print 'Flipping images...'
    flip_images( root_path, 'test' )
    flip_images( root_path, 'train' )

    print 'Rotating images...'
    rotate_and_translate( root_path, 'test' )
    rotate_and_translate( root_path, 'train' )

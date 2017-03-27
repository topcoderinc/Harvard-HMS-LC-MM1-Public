#!/usr/bin/env python
from libs.hms_lung_tumor import *

if __name__ == "__main__":
    # Path to patient files
    print 'Calculating image mean...'

    root_path = '/home/thomio/datasets/lung_tumor/example_extracted'
    compute_images_mean( root_path )

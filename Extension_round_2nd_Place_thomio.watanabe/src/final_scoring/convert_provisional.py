#!/usr/bin/env python
from hms_lung_tumor import HMSLungTumor
import argparse

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, required=True)
args = parser.parse_args()

root_path = args.root_path

if __name__ == "__main__":
    gt = HMSLungTumor()

    print 'Pre-processing testing images...'
    gt.convert_provisional_images( root_path )

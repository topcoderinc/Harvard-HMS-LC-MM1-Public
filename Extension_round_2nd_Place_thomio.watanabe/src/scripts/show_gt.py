#!/usr/bin/env python
from libs.hms_lung_tumor import *

if __name__ == "__main__":

#    scan_path = '/home/thomio/datasets/lung_tumor/example_extracted/test/ANON_LUNG_TC040'
#    scan_path = '/home/thomio/datasets/lung_tumor/example_extracted/test/ANON_LUNG_TC066'
#    scan_path = '/home/thomio/datasets/lung_tumor/example_extracted/test/ANON_LUNG_TC079'
    scan_path = '/home/thomio/datasets/lung_tumor/example_extracted/test/ANON_LUNG_TC393'
#    scan_path = '/home/thomio/datasets/lung_tumor/example_extracted/test/ANON_LUNG_TC450'
#    scan_path = '/home/thomio/datasets/lung_tumor/example_extracted/test/ANON_LUNG_TC506'
#    scan_path = '/home/thomio/datasets/lung_tumor/example_extracted/test/ANON_LUNG_TC541'
#    scan_path = '/home/thomio/datasets/lung_tumor/example_extracted/test/ANON_LUNG_TC589'
    show_gt( scan_path )

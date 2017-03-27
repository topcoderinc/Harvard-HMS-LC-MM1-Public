#!/usr/bin/env python
import numpy as np
import sys
import cv2
import os

import hms_lung_tumor as hms


def save_imgs_paths( root_path, files_type ):
    text_file_name = 'class_' + files_type + '.txt'
    text_file = open( text_file_name, 'w')
    balanced_file = open( 'class_balanced_' + files_type + '.txt', 'w')

    all_scans_ids = os.listdir( root_path + files_type )
    for scan_id in all_scans_ids:
        path_to_files = root_path + files_type + '/' + scan_id
        print path_to_files

        tumor_numbers = hms.read_structures_file( path_to_files, 'radiomics_gtv' )

        images_files = os.listdir( path_to_files + '/pngs/' )
        for image_name in images_files:
            # Remove .png
            image_name = image_name.split('.')
            image_name = image_name[0]

            contours = []
            for tumor_number in tumor_numbers:
                file_path = path_to_files + '/contours/' + image_name.lstrip('0') + '.' + str(tumor_number) + '.dat'
                if os.path.exists( file_path ):
                    contours = hms.read_contour( path_to_files, image_name, str(tumor_number), contours )

            msg = files_type + '/' + scan_id + '/pngs/' + image_name + '.png'
            msg = msg + ' '

            skip = False
            if( contours ):
                msg = msg + '1\n'
            else:
                skip = True
                msg = msg + '0\n'
                if( np.random.randint(6, size=1) == 3 ):
                    skip = False

            text_file.write( msg )
            if( not skip ):
                balanced_file.write( msg )

    text_file.close()
    balanced_file.close()
    return

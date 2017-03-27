#!/usr/bin/env python
from libs.hms_lung_tumor import *
import numpy as np
import shutil
import sys
import cv2
import os



def convert_images( root_path, files_type ):
    all_scans_ids = os.listdir( root_path + files_type )
    for scan_id in all_scans_ids:
        path_to_files = root_path + files_type + '/' + scan_id
        print path_to_files
        # content in path_to_files:
        # auxiliary/
        # contours/
        # ground_truth/ -> is going to be created
        # pngs/
        # structures.dat

        images_files = os.listdir( path_to_files + '/pngs/' )
        for image_name in images_files:
            path_to_img = path_to_files + '/pngs/' + image_name
            img = cv2.imread( path_to_img, cv2.IMREAD_GRAYSCALE )

            height, width = img.shape
            if (height == 512 and width == 512):
                # Convert image to float point
                img = img.astype( np.float )

                max_value = np.amax( img )
                # Rescale image intensity
                img = (img / max_value) * 255
                img = img.astype( np.uint8 )

                # Crop image in:
                # row: before 100, after 360
                # col: before 100, after 420
                img = img[100:360, 100:420]

                cv2.imwrite( path_to_img, img )
            else:
                print 'Error: input image size != 512x512'
                print path_to_img
                sys.exit(0)
    return


def save_path_to_files( root_path, files_type, scan_id, file_name, images_type = ''):
    if( images_type ):
        text_file_name = files_type + '_' + images_type + '.txt'
    else:
        text_file_name = files_type + '.txt'

    text_file = open( text_file_name, 'a')

    msg = root_path + files_type + '/' + scan_id + '/pngs/' + file_name + '.png'
    msg = msg + ' '
    msg = msg + root_path + files_type + '/' + scan_id + '/ground_truth/' + file_name + '.png'
    msg = msg + '\n'

    text_file.write( msg )
    return


def save_file_with_imgs_paths( root_path, files_type ):
    all_scans = os.listdir( root_path + files_type )
    for scan_id in all_scans:
        path_to_scan = root_path + files_type + '/' + scan_id
        print path_to_scan

        tumor_numbers = read_structures_file( path_to_scan, 'radiomics_gtv' )

        images_files = os.listdir( path_to_scan + '/pngs/' )
        for image_name in images_files:
            # Remove .png
            image_name = image_name.split('.')
            image_name = image_name[0]

            contours = []
            for tumor_number in tumor_numbers:
                file_path = path_to_scan + '/contours/' + image_name.lstrip('0') + '.' + str(tumor_number) + '.dat'
                if os.path.exists( file_path ):
                    contours = read_contour( path_to_scan, image_name, str(tumor_number), contours )

            save_path_to_files( root_path, files_type, scan_id, image_name )

            if( contours ):
                save_path_to_files( root_path, files_type, scan_id, image_name, 'tumors_only' )
            elif( np.random.randint(2, size=1) == 1 ):
                save_path_to_files( root_path, files_type, scan_id, image_name, 'no_tumors' )

            if( contours ):
                save_path_to_files( root_path, files_type, scan_id, image_name, 'mix' )
            elif( np.random.randint(7, size=1) == 3 ):
                save_path_to_files( root_path, files_type, scan_id, image_name, 'mix' )
    return


def convert_provisional_images( root_path ):
    all_scans = os.listdir( root_path )
    for scan_id in all_scans:
        path_to_scan = root_path + scan_id
        print path_to_scan
        # content in path_to_scan:
        # auxiliary/
        # pngs/
        images_files = os.listdir( path_to_scan + '/pngs/' )
        for file_name in images_files:
            path_to_img = path_to_scan + '/pngs/' + file_name
            img = cv2.imread( path_to_img, cv2.IMREAD_GRAYSCALE )

            height, width = img.shape
            if (height == 512 and width == 512):
                # Convert image to float point
                img = img.astype( np.float )

                max_value = np.amax( img )
                # Rescale image intensity
                img = (img / max_value) * 255
                img = img.astype( np.uint8 )

                img = img[100:360, 100:420]

                cv2.imwrite( path_to_img, img )
            else:
                print 'Error: input image size != 512x512'
                print path_to_img
                sys.exit(0)
    return


def save_path_to_inference( root_path ):
    text_file_name = 'inference.txt'
    text_file = open( text_file_name, 'w')

    all_scans = os.listdir( root_path )
    for scan_id in all_scans:
        path_to_scan = root_path + scan_id
        print path_to_scan

        images_files = os.listdir( path_to_scan + '/pngs/' )
        for image_name in images_files:
            msg = path_to_scan + '/pngs/' + image_name
            msg = msg + ' '
            msg = msg + path_to_scan + '/pngs/' + image_name
            msg = msg + '\n'

            text_file.write( msg )
    return


def rename_images( root_path, files_type ):
    all_scans_ids = os.listdir( root_path + files_type )
    for scan_id in all_scans_ids:
        path_to_scan = root_path + files_type + '/' + scan_id
        print path_to_scan

        all_images = os.listdir( path_to_scan + '/pngs/' )
        for image_name in all_images:
            # Remove .png
            slice_number = image_name.split('.')[0]
            slice_id = slice_number.rjust(5, '0')

            if len(slice_number) != 5:
                path_to_image = path_to_scan + '/pngs/' + image_name
                image = cv2.imread( path_to_image, cv2.IMREAD_GRAYSCALE )

                cv2.imwrite( path_to_scan + '/pngs/' + slice_id + '.png', image )

                if os.path.exists( path_to_image ):
                    os.remove( path_to_image )
    return


def concat_images( root_path, files_type ):
    all_scans = os.listdir( root_path + files_type )
    for scan_id in all_scans:
        path_to_scan = root_path + files_type + '/' + scan_id
        print path_to_scan

        concat_directory = path_to_scan + '/concat_images'
        if os.path.exists( concat_directory ):
            shutil.rmtree( concat_directory )
        os.makedirs( concat_directory + '/pngs/')

        all_images = os.listdir( path_to_scan + '/pngs/' )
        all_images = sorted( all_images )

        for i in range( len(all_images) - 2 ):
            path_to_image_01 = path_to_scan + '/pngs/' + all_images[ i ]
            path_to_image_02 = path_to_scan + '/pngs/' + all_images[ i + 1]
            path_to_image_03 = path_to_scan + '/pngs/' + all_images[ i + 2]

            image_01 = cv2.imread( path_to_image_01, cv2.IMREAD_GRAYSCALE )
            image_02 = cv2.imread( path_to_image_02, cv2.IMREAD_GRAYSCALE )
            image_03 = cv2.imread( path_to_image_03, cv2.IMREAD_GRAYSCALE )

            line, col = image_01.shape
            image_01 = image_01.reshape(line, col, 1);
            image_02 = image_02.reshape(line, col, 1);
            image_03 = image_03.reshape(line, col, 1);


            result = np.concatenate(( image_01, image_02, image_03), axis = 2 )
#            print 'Shape = ', result.shape
#            print 'Size = ', result.size
#            print 'Data type = ', result.dtype
#            print 

            cv2.imwrite( concat_directory + '/pngs/' + all_images[i + 1], result )
    return


def save_path_to_concat( root_path, files_type ):
    text_file_name_01 = files_type + '_concat.txt'
    text_file_01 = open( text_file_name_01, 'w')

    text_file_name_02 = files_type + '_concat_tumors_only.txt'
    text_file_02 = open( text_file_name_02, 'w')

    all_scans = os.listdir( root_path + files_type )
    for scan_id in all_scans:
        path_to_scan = root_path + files_type + '/'+ scan_id
        print path_to_scan

        concat_directory = path_to_scan + '/concat_images/pngs/'
        if not os.path.exists( concat_directory ):
            print "Error. concat_images/pngs/ directory doesn't exist !!!"
            sys.exit()

        tumor_numbers = read_structures_file( path_to_scan, 'radiomics_gtv' )

        images_files = os.listdir( concat_directory )
        for image_name in images_files:

            slice_id = image_name.split('.')[0]
            contours = []
            for tumor_number in tumor_numbers:
                file_path = path_to_scan + '/contours/' + slice_id.lstrip('0') + '.' + str(tumor_number) + '.dat'
                if os.path.exists( file_path ):
                    contours = read_contour( path_to_scan, slice_id, str(tumor_number), contours )

            msg = concat_directory + image_name
            msg = msg + ' '
            msg = msg + path_to_scan + '/ground_truth/' + image_name
            msg = msg + '\n'

            text_file_01.write( msg )
            if( contours ):
                text_file_02.write( msg )

    text_file_01.close()
    text_file_02.close()
    return



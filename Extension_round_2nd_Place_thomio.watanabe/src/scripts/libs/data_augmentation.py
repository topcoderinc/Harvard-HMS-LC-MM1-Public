#!/usr/bin/env python
import numpy as np
import random
import shutil
import sys
import cv2
import os


def flip_images( root_path, files_type ):
    all_scans_ids = os.listdir( root_path + files_type )
    for scan_id in all_scans_ids:
        path_to_scan = root_path + files_type + '/' + scan_id
        print path_to_scan

        path_to_gtruth = path_to_scan + '/ground_truth/'
        if not os.path.exists( path_to_gtruth ):
            print 'Error: ground truth directory does not exist.'
            print 'Scan:', path_to_scan
            sys.exit()

        flipped_directory = path_to_scan + '/flipped/'
        if os.path.exists( flipped_directory + '/pngs/' ):
            shutil.rmtree( flipped_directory )
        if os.path.exists( flipped_directory + '/ground_truth/' ):
            shutil.rmtree( flipped_directory )

        os.makedirs( flipped_directory + '/pngs/' )
        os.makedirs( flipped_directory + '/ground_truth/' )


        all_images = os.listdir( path_to_scan + '/pngs/' )
        for image_name in all_images:
            path_to_image = path_to_scan + '/pngs/' + image_name
            path_to_gtruth = path_to_scan + '/ground_truth/' + image_name

            image = cv2.imread( path_to_image )
            gtruth = cv2.imread( path_to_gtruth, cv2.IMREAD_GRAYSCALE )

            rows, cols, chs = image.shape

            # Add gaussian noise
            noise = np.random.normal( 5, 2, [rows, cols, chs] )
            noise *= (noise>0)
            noise = np.round( noise ).astype( dtype=np.uint8)
            noisy_image = image + noise

            # flip horizontaly or verticaly randonly
            orientation = np.random.randint(2, size=1)
            # 0 = horizontal flip-> upside-down
            # 1 = vetical flip
            flipped_image = cv2.flip(noisy_image, orientation)
            flipped_gtruth = cv2.flip(gtruth, orientation)

            # Rotate image
            rotation = random.sample(set([-25, -17, -12, -8, 2, 7, 17, 22]), 1)

            M = cv2.getRotationMatrix2D( (cols/2,rows/2), rotation[0], 1 )
            rotated_image = cv2.warpAffine(flipped_image, M, (cols,rows))

            M = cv2.getRotationMatrix2D( (cols/2,rows/2), rotation[0], 1 )
            rotated_gtruth = cv2.warpAffine(flipped_gtruth, M, (cols,rows))

            # Trnaslate image
            h_shift = random.sample(set([-53, -44, -37, -22, -14, -5, 9, 11, 24, 33, 42]), 1)
            v_shift = random.sample(set([-20, 18, -12, -8, 4, 9, 15, 22, 26]), 1)

            M = np.float32([[1,0,h_shift[0]],[0,1,v_shift[0]]])
            translated_image = cv2.warpAffine(rotated_image,M,(cols,rows))

            M = np.float32([[1,0,h_shift[0]],[0,1,v_shift[0]]])
            translated_gtruth = cv2.warpAffine(rotated_gtruth,M,(cols,rows))


            cv2.imwrite( flipped_directory + '/pngs/' + image_name, translated_image )
            cv2.imwrite( flipped_directory + '/ground_truth/' + image_name, translated_gtruth )
    return



def rotate_and_translate( root_path, files_type ):
    all_scans_ids = os.listdir( root_path + files_type )
    for scan_id in all_scans_ids:
        path_to_scan = root_path + files_type + '/' + scan_id
        print path_to_scan

        path_to_gtruth = path_to_scan + '/ground_truth/'
        if not os.path.exists( path_to_gtruth ):
            print 'Error: ground truth directory does not exist.'
            print 'Scan:', path_to_scan
            sys.exit()

        rotated_directory = path_to_scan + '/rotated/'
        if os.path.exists( rotated_directory + '/pngs/' ):
            shutil.rmtree( rotated_directory )
        if os.path.exists( rotated_directory + '/ground_truth/' ):
            shutil.rmtree( rotated_directory )

        os.makedirs( rotated_directory + '/pngs/' )
        os.makedirs( rotated_directory + '/ground_truth/' )


        all_images = os.listdir( path_to_scan + '/pngs/' )
        for image_name in all_images:
            path_to_image = path_to_scan + '/pngs/' + image_name
            path_to_gtruth = path_to_scan + '/ground_truth/' + image_name

            image = cv2.imread( path_to_image  )
            gtruth = cv2.imread( path_to_gtruth, cv2.IMREAD_GRAYSCALE )

            rows, cols, chs = image.shape

            # Add gaussian noise
            noise = np.random.normal( 5, 2, [rows, cols, chs] )
            noise *= (noise>0)
            noise = np.round( noise ).astype( dtype=np.uint8)
            noisy_image = image + noise

            # Rotate image
            rotation = random.sample(set([-19, -14, -11, 4, 17, 25]), 1)

            M = cv2.getRotationMatrix2D( (cols/2,rows/2), rotation[0], 1 )
            rotated_image = cv2.warpAffine(noisy_image, M, (cols,rows))

            M = cv2.getRotationMatrix2D( (cols/2,rows/2), rotation[0], 1 )
            rotated_gtruth = cv2.warpAffine(gtruth, M, (cols,rows))

            # Trnaslate image
            h_shift = random.sample(set([-52, -41, -27, -15, 8, 19, 25, 34, 45, 53]), 1)
            v_shift = random.sample(set([-20, -14, -11, -4, 0, 7, 10, 16, 21]), 1)

            M = np.float32([[1,0,h_shift[0]],[0,1,v_shift[0]]])
            translated_image = cv2.warpAffine(rotated_image,M,(cols,rows))

            M = np.float32([[1,0,h_shift[0]],[0,1,v_shift[0]]])
            translated_gtruth = cv2.warpAffine(rotated_gtruth,M,(cols,rows))


            cv2.imwrite( rotated_directory + '/pngs/' + image_name, translated_image )
            cv2.imwrite( rotated_directory + '/ground_truth/' + image_name, translated_gtruth )
    return


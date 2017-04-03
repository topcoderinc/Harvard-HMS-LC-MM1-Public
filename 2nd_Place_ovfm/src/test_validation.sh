#!/bin/bash

LUNG_CHECKPOINT=$1
WORK_DIR=$2

python train.py --data_dir "/home/ubuntu/data/validation_data/data-" --model_dir $WORK_DIR/dummy --batch_size 4 \
  --checkpoint $LUNG_CHECKPOINT --validate_output_dir $WORK_DIR/predictions/validation_lungs --model_variation lung

python slice_lung.py --input_dir /home/ubuntu/data/validation_raw --predictions_dir $WORK_DIR/predictions/validation_lungs \
  --output_directory $WORK_DIR/lung_validation_examples

python create_records.py --input_dir $WORK_DIR/lung_validation_examples --input_annotations /home/ubuntu/data/validation_annotations.txt \
  --output_directory $WORK_DIR/validation_data_radiomics --lungs

python train.py --data_dir $WORK_DIR/validation_data_radiomics/data- --model_dir $WORK_DIR/dummy_2 --batch_size 12 \
  --model_variation radiomics --checkpoint "/home/ubuntu/data/models/segmentation/radiomics_vgg/model-16000" \
  --validate_output_dir $WORK_DIR/predictions/validation_radiomics 

python generate_contours.py --predictions_dir $WORK_DIR/predictions/validation_radiomics \
  --input_annotations /home/ubuntu/data/validation_annotations.txt \
  --input_lungs_dir $WORK_DIR/lung_validation_examples --input_dir /home/ubuntu/data/validation_raw --output_contours_path $WORK_DIR/output_validate_contour 

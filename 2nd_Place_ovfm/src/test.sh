#!/bin/bash
TEST_EXAMPLES_DIR=$1
MODELS_DIR=$2
PREDICTIONS_FILE=$3
WORK_DIR=/dev/workdir
source $HOME/venv/bin/activate

rm -rf $WORK_DIR
mkdir -p $WORK_DIR

python convert_to_dataset.py --input_dir $TEST_EXAMPLES_DIR --output_file $WORK_DIR/test_annotations.txt   
python create_records.py --input_dir $TEST_EXAMPLES_DIR --input_annotations $WORK_DIR/test_annotations.txt \
  --output_directory $WORK_DIR/test_data

python train.py --data_dir $WORK_DIR/test_data/data- --model_dir $WORK_DIR/models/lung --batch_size 4 \
  --checkpoint $MODELS_DIR/model-28000 --validate_output_dir $WORK_DIR/predictions/test_lungs

python slice_lung.py --input_dir $TEST_EXAMPLES_DIR --predictions_dir $WORK_DIR/predictions/test_lungs \
  --output_directory $WORK_DIR/lung_test_examples

python create_records.py --input_dir $WORK_DIR/lung_test_examples --input_annotations $WORK_DIR/test_annotations.txt \
  --output_directory $WORK_DIR/test_data_radiomics --lungs

python train.py --data_dir $WORK_DIR/test_data_radiomics/data --model_dir $WORK_DIR/models/radiomics --batch_size 12 \
  --model_variation radiomics --checkpoint $MODELS_DIR/model-16000 \
  --validate_output_dir $WORK_DIR/predictions/test_radiomics 

python generate_contours.py --predictions_dir $WORK_DIR/predictions/test_radiomics --input_annotations $WORK_DIR/test_annotations.txt \
  --input_lungs_dir $WORK_DIR/lung_test_examples --input_dir $TEST_EXAMPLES_DIR --output_contours_path $PREDICTIONS_FILE 


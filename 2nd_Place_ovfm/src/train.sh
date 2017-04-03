#!/bin/bash
TRAIN_EXAMPLES_DIR=$1
MODELS_DIR=$2
WORK_DIR=workdir
source $HOME/hms/bin/activate

python convert_to_dataset.py --input_dir $TRAIN_EXAMPLES_DIR --output_file $WORK_DIR/train_annotations.txt   
python create_records.py --input_dir $TRAIN_EXAMPLES_DIR --input_annotations $WORK_DIR/train_annotations.txt \
  --output_directory $WORK_DIR/train_data

python train.py --data_dir $WORK_DIR/train_data --model_dir $WORK_DIR/models/lungs --batch_size 4 --max_iters 30000 \
  --model_variation lung
python train.py --data_dir $WORK_DIR/train_data --model_dir $WORK_DIR/models/lungs --batch_size 4 --max_iters 30000 \
  --checkpoint $WORK_DIR/models/lungs/model-30000 --validate_output_dir $WORK_DIR/predictions/train_lungs

python slice_lung.py --input_dir $TRAIN_EXAMPLES_DIR --predictions_dir $WORK_DIR/predictions/train_lungs \
  --output_directory $WORK_DIR/lung_train_examples
python create_records.py --input_dir $TRAIN_EXAMPLES_DIR --input_annotations $WORK_DIR/train_annotations.txt \
  --output_directory $WORK_DIR/train_data_radiomics --lungs

python train.py --data_dir $WORK_DIR/train_data_radiomics --model_dir $WORK_DIR/models/radiomics_vgg --batch_size 12 \
  --max_iters 20000 --model_variation radiomics_vgg

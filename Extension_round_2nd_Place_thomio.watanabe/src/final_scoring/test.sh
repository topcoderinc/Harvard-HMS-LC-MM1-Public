#!/bin/bash

# ./test.sh /home/thomio/datasets/lung_tumor/example_extracted/test/ ./SegNet-Tutorial/Models/Inference/lungnet_weights.caffemodel output_file.csv

if [ $# -ne 3 ]
then
  echo 'Usage: test.sh /path/to/test/dataset/ /path/to/model/model.caffemodel output_file.csv'
  exit 1
fi

echo 'Path to dataset =' $1
echo 'Path to model =' $2
echo 'Output file name =' $3
echo 


# NOTE: the pre-processing is required only once
# You can comment this command later (but not delete the test.txt file).
# echo 'Pre-processing testing images...'
python /home/thomio/sandbox/lung_cancer/final_scoring/convert_provisional.py --root_path $1
echo


# echo 'Generating caffe inference.txt file...'
python /home/thomio/sandbox/lung_cancer/final_scoring/generate_test_file.py --root_path $1
echo


# echo 'Running inference on test dataset...'
python /home/thomio/sandbox/lung_cancer/final_scoring/test_segmentation.py \
        --model /home/thomio/sandbox/lung_cancer/caffe/models/lungnet_basic_inference.prototxt \
        --weights $2 \
        --root_path $1 \
        --text_file_name inference.txt
echo


# echo 'Generating output file...'
python /home/thomio/sandbox/lung_cancer/final_scoring/generate_bb.py --root_path $1 --output_file $3

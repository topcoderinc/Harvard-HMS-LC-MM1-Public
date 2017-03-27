Thomio Watanabe    
February 2017    

TopCoder    
Contest: HMS Lung Cancer    
Problem: [LungTumorTracer](https://community.topcoder.com/longcontest/?module=ViewProblemStatement&compid=54489&rd=16870)    


input image size after crop -> 260 x 320


Train set
---------
Patients scans: ./data_selection/dirs_train.txt

Total number of images = 34322    
Caffe training images: ./caffe/train.txt    

Images with cancer = 3645 (10,62%)    
Caffe training images: ./caffe/train_tumors_only.txt    

train_mix_01.txt ->  = 3645 (tumor) +  (random empty)    
train_mix_02.txt ->  = 3645 (tumor) +  (random empty)    



Test set
--------
Patients scans: ./data_selection/dirs_train.txt    

Total number of images = 4591    
Caffe training images: ./caffe/train.txt    

Images with cancer = 455 (9,91%)    
Caffe training images: ./caffe/train_only_tumor_imgs.txt    

test_mix_01.txt ->  = 455 (tumor) +  (random empty)    
test_mix_02.txt ->  = 455 (tumor) +  (random empty)    


Train + Test set
----------------
all_mix_01.txt -> train_mix_01 + test_mix_01 -> 9690 images    
all_mix_01.txt -> train_mix_01 + test_mix_01 -> 11078 images    



Files with problems
-------------------
Removed from caffe training images:    
/home/thomio/datasets/lung_tumor/example_extracted/train/ANON_LUNG_TC470/pngs/114.png    



STEPS
-----
1 - Separate the images in train and test ( see data_selection/ )    
2 - Convert images and generate ground truth
    - scripts/rename_images.py
    - scripts/convert_images.py
    - cripts/generate_gt.py    
3 - Train caffe model    
4 - Adjust the BN weights    
5 - Convert provisional images -> scripts/convert_provisional.py (if was not done before)    
6 - Run inference    
7 - Generate the bouding boxes (result.cvs file) -> scripts/generate_bb.py    



COMMANDS
--------
Train model:    
./caffe-segnet/build/tools/caffe train --solver ./Models/lungnet_basic_solver.prototxt    

Train model with weights:    
./caffe-segnet/build/tools/caffe train --solver ./Models/lungnet_basic_solver.prototxt --snapshot ./Models/Training/lungnet_basic_iter_1000.solverstate    


BN weights:    
python ./Scripts/compute_bn_statistics.py ./Models/lungnet_train.prototxt ./Models/Training/lungnet_iter_100000.caffemodel    
mv ./Models/Inference/lungnet_weights.caffemodel ./Models/Inference/06_lungnet_weights.caffemodel    
rm ./Models/Inference/__for_calculating_BN_stats_lungnet_train.prototxt    

Run inference:    
python ./Scripts/test_segmentation.py --model ./Models/lungnet_inference.prototxt --weights ./Models/Inference/lungnet_weights.caffemodel    




WEIGHTS
-------

train_mix_02.csv
class_weighting: 0.503600934887
class_weighting (tumor): 69.92641504

train.txt / all_images.txt
class_weighting: 0.501023752384
class_weighting (tumor): 244.699675469

train_lung_gt.txt
Normal cells weight =  0.0922664783029
Lung cells weight =  1.0
Cancer cells weight =  43.5374927863

train_lung_gt_mix.txt
Normal cells weight =  0.172012375032
Lung cells weight =  1
Cancer cells weight =  23.9366008022



lUNG CANCER IMAGE CLASSIFICATION
--------------------------------

1 - Generate caffe training files ->  class_generate_train_files.py
2 - Create lmdb datasets

./caffe-segnet/build/tools/convert_imageset --gray --shuffle \
    /home/thomio/datasets/lung_tumor/example_extracted/ \
    /home/thomio/sandbox/lung_cancer/caffe/classification/class_train.txt \
    /home/thomio/datasets/lung_tumor/data/lungnet_train_lmdb

./caffe-segnet/build/tools/convert_imageset --gray --shuffle \
    /home/thomio/datasets/lung_tumor/example_extracted/ \
    /home/thomio/sandbox/lung_cancer/caffe/classification/class_test.txt \
    /home/thomio/datasets/lung_tumor/data/lungnet_test_lmdb

./caffe-segnet/build/tools/convert_imageset --gray --shuffle \
    /home/thomio/datasets/lung_tumor/example_extracted/ \
    /home/thomio/sandbox/lung_cancer/caffe/classification/class_val.txt \
    /home/thomio/datasets/lung_tumor/data/lungnet_val_lmdb


./caffe-segnet/build/tools/convert_imageset --gray --shuffle \
    /home/thomio/datasets/lung_tumor/example_extracted/ \
    /home/thomio/sandbox/lung_cancer/caffe/classification/class_balanced_train_aug.txt \
    /home/thomio/datasets/lung_tumor/data/lungnet_balanced_train_lmdb

./caffe-segnet/build/tools/convert_imageset --gray --shuffle \
    /home/thomio/datasets/lung_tumor/example_extracted/ \
    /home/thomio/sandbox/lung_cancer/caffe/classification/class_balanced_test_aug.txt \
    /home/thomio/datasets/lung_tumor/data/lungnet_balanced_test_lmdb

./caffe-segnet/build/tools/convert_imageset --gray --shuffle \
    /home/thomio/datasets/lung_tumor/example_extracted/ \
    /home/thomio/sandbox/lung_cancer/caffe/classification/class_balanced_train_all.txt \
    /home/thomio/datasets/lung_tumor/data/lungnet_balanced_all_lmdb



Avaliar:
 - centralizar o paciente antes de corta-las *** mais importante ***
    - erosão e dilatação das imagens
    - transformar em images preto/branco 0/255
    
 - Aumentar as imagens
 - Sobreposição dos BB no resultado
 - Modelos maiores de segmentação: deconvnet, etc




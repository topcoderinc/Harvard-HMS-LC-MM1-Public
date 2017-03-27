Thomio Watanabe    
February 2017    

TopCoder    
Contest: HMS Lung Cancer    
Problem: [LungTumorTracer](https://community.topcoder.com/longcontest/?module=ViewProblemStatement&compid=54489&rd=16870)    


input image size after crop -> 260 x 320


Train set
---------
Patients scans: ./data_selection/dirs_train.txt

Total number of images = 34321    
Caffe training images: ./caffe/train.txt    

Images with cancer = 4846 (14,1196%)    
Caffe training images: ./caffe/train_tumors_only.txt    

train_mix_01.txt -> 8550 = 4846 (tumor) + 3704 (random empty)    
train_mix_02.txt -> 9808 = 4846 (tumor) + 4962 (random empty)    



Test set
--------
Patients scans: ./data_selection/dirs_train.txt    

Total number of images = 4591    
Caffe training images: ./caffe/train.txt    

Images with cancer = 613 (13,352%)    
Caffe training images: ./caffe/train_only_tumor_imgs.txt    

test_mix_01.txt -> 1140 = 613 (tumor) + 527 (random empty)    
test_mix_02.txt -> 1270 = 613 (tumor) + 657 (random empty)    


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
2 - Convert images and generate ground truth -> scripts/generate_gt.py    
3 - Train caffe model    
4 - Adjust the BN weights    
5 - Convert provisional images -> scripts/convert_provisional.py (if was not done before)    
6 - Run inference    
7 - Generate the bouding boxes (result.cvs file) -> scripts/generate_bb.py    



COMMANDS
--------
Train model:    
./caffe-segnet/build/tools/caffe train -solver ./Models/lungnet_basic_solver.prototxt    

Train model with weights:    
./caffe-segnet/build/tools/caffe train -solver ./Models/lungnet_basic_solver.prototxt -weights ./Models/Training/lungnet_basic_iter_1000.caffemodel    

BN weights:    
mv ./Models/Inference/lungnet_weights.caffemodel ./Models/Inference/06_lungnet_weights.caffemodel    
rm ./Models/Inference/__for_calculating_BN_stats_lungnet_basic_train.prototxt ./Models/Inference/test_weights.caffemodel    
python ./Scripts/compute_bn_statistics.py ./Models/lungnet_basic_train.prototxt ./Models/Training/lungnet_basic_iter_1000.caffemodel ./Models/Inference/    
cp ./Models/Inference/test_weights.caffemodel ./Models/Inference/lungnet_weights.caffemodel    

Run inference:    
python ./Scripts/test_segmentation.py --model ./Models/lungnet_basic_inference.prototxt --weights ./Models/Inference/lungnet_weights.caffemodel    


BN weights:    
mv ./Models/Inference/large_lungnet_weights.caffemodel ./Models/Inference/01_large_lungnet_weights.caffemodel
rm ./Models/Inference/__for_calculating_BN_stats_large_lungnet_basic_train.prototxt ./Models/Inference/test_weights.caffemodel    
python ./Scripts/compute_bn_statistics.py ./Models/large_lungnet_basic_train.prototxt ./Models/Training/large_lungnet_basic_iter_1000.caffemodel ./Models/Inference/    
cp ./Models/Inference/test_weights.caffemodel ./Models/Inference/large_lungnet_weights.caffemodel    

Run inference:    
python ./Scripts/test_segmentation.py --model ./Models/large_lungnet_basic_inference.prototxt --weights ./Models/Inference/large_lungnet_weights.caffemodel    
python ./Scripts/3ch_test_segmentation.py --model ./Models/large_lungnet_basic_inference.prototxt --weights ./Models/Inference/large_lungnet_weights.caffemodel    



MISC
----
crop image in:    
row: before 100, after 360    
col: before 100, after 420    


03_results.csv -> 135606.82


05_lungnet_weights.caffemodel
04_test.csv -> 112083.652031
04_results.csv -> 155104.27

06_lungnet_weights.caffemodel
05_test.csv -> 124875.747368
05_results.csv -> 168725.92	

07_lungnet_weights.caffemodel
06_test.csv -> 52028.663378
06_results.csv -> ?

01_tumors_lungnet_weights.caffemodel
07_test.csv -> 8239.956856
07_results.csv -> ?


results/croped/08_results.csv -> 185295.79
results/croped/09_results.csv -> 160079.51
results/croped/10_results.csv -> 165025.36


07_lungnet_weights.caffemodel (mix images ?)
results/croped/06_test.csv -> 127609.483395
results/croped/results.csv -> not submitted

08_lungnet_weights.caffemodel (mix images ?)
results/croped/07_test.csv -> 130068.099067
results/croped/results.csv -> not submitted

09_15000_lungnet_weights.caffemodel (all images)
drive - final/09_15000_results.csv -> 169573.61(verified)

09_lungnet_weights.caffemodel (all images)
results/croped/test.csv -> 245525.881016
drive - final/09_results.csv -> 185295.79(verified)

10_lungnet_weights.caffemodel (all images)
results/croped/test.csv -> 266916.283756
drive - final/10_results.csv -> 160079.51(verified)

11_lungnet_weights.caffemodel (all images)
results/croped/test.csv -> 295733.928436
results/croped/results.csv -> 


results/deep/3ch_01_test.csv -> 132277.907 (lung_mix)
results/deep/01_test.csv -> 103347.653246 (mix_01)
results/deep/02_test_tumors_only.csv -> 126226.179448
results/deep/03_test_tumors_only.csv -> 131257.666992  (90%)
results/deep/04_test.csv -> 129492.073897 (mix_01) (85%-90%)
results/deep/05_test.csv -> 137376.469837 (lung_mix_01) (85%-90%)

final_results -> 154141.77
3ch_final_results -> 163631.40



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



Possible improvements
---------------------
- usar o modelo completo do segnet    

- remover a média das imagens    
- segmentar o pulmão    



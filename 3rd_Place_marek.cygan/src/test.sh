if [ $# -ne 3 ]
then
  echo Usage ./test.sh test_dir model_folder output_file
  exit 1
fi

testdir=$1
modeldir=$2
outfile=$3

python process.py --path $testdir
cp scans.csv scans_test.csv
python -u main.py --dilarch 1 --batch_size 32 --arch_multiplier 1.5 --crop 300 --final 300 --aug_size 20 --rotate 10 --restore_path $modeldir/model3/final.ckpt --means $modeldir/model3/means.p --predict '001' --pred_iter 10 --test_dir $testdir
ls predictions/test/*_0.jpg > test_list.txt
./vis --predictions test_list.txt 109
cp submission.csv $outfile

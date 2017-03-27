for i in `seq 0 1000`
do
  if [ -f runs/$i.log ] 
  then
    continue
  else
    runid=$i
    break
  fi
done

args=$*
echo $args
echo Runid=$runid

mkdir runs/$runid
mkdir runs/$runid/models
cp *.py runs/$runid

echo `date` >> runs/$runid/date.txt

com="python -u main.py $args --models_dir runs/$runid/models --means_store runs/$runid/means.p"
echo $com > runs/$runid/command.txt

$com |& tee runs/${runid}.log

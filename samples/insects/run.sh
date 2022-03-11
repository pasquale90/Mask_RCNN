#!bin/bash
ps aux |grep nohup

dt=`date "+%Y%m%d%H%M"`
nohup_output="logs/${dt}_train_insects_nohup-out.txt"
nohup bash train_insects.sh  > ${nohup_output} & 

#nohup_output="logs/${dt}_example_nohup-out.txt"
#nohup bash example.sh  > ${nohup_output} & 
tail -f ${nohup_output}
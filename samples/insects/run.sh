#!bin/bash
ps aux |grep nohup

dt=`date "+%Y%m%d%H%M"`

nohup_output="logs/${dt}_train_plastelines_nohup-out.txt"
#nohup bash samples/insects/train_insects.sh train datasets/Nasekomo_insects phase1_Nasekomo_3609.txt pretrained_models/mask_rcnn_balloon.h5 > ${nohup_output} & 
nohup bash samples/insects/train_insects.sh train datasets/plastelines2 plastelines_annotations.txt pretrained_models/mask_rcnn_balloon.h5 > ${nohup_output} & 
tail -f ${nohup_output}

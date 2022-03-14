#!bin/bash

#initializing python
source /home/melissap/miniconda3/etc/profile.d/conda.sh
conda activate mask

python3 train_insects.py train --dataset=/data/CoRoSect/10.code/maskRCNN/Mask_RCNN_matterport/datasets/Nasekomo_insects --weights=/data/CoRoSect/10.code/maskRCNN/Mask_RCNN_matterport/mask_rcnn_balloon.h5

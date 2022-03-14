#!bin/bash

#initializing python
source /home/melissap/miniconda3/etc/profile.d/conda.sh
conda activate mask

python3 train_insects.py train --dataset=datasets/Nasekomo_insects --weights=mask_rcnn_balloon.h5

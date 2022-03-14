#!bin/bash

#initializing python
source /home/melissap/miniconda3/etc/profile.d/conda.sh
conda activate mask

python3 samples/insects/train_insects.py $1 --dataset=$2 --ImaggeTagger_file=$3 --weights=$4




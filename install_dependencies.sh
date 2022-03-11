#!bin/bash
cwd=$PWD
echo $cwd
cd ~/miniconda3/envs/mask/
./bin/pip install -r $cwd/my_requirements.txt

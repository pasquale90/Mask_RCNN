#!bin/bash
cwd=$PWD
echo $cwd

echo "creating a new conda evn named mask"
bash restore-conda-envs.sh pending

cd ~/miniconda3/envs/mask/
./bin/pip install -r $cwd/my_requirements.txt

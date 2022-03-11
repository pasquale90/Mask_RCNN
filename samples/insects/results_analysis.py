import datetime
%reload_ext tensorboard
log_directory="/data/CoRoSect/10.code/maskRCNN/Mask_RCNN_matterport/logs/insects20220214T1132"
%tensorboard --logdir log_directory --port=8008 --host=127.0.0.1
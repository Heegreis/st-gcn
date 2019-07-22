import os
import time

tStart = time.time()

# --model net.st_gcn.Model --work_dir ./work_dir/recognition/ntu-xsub/ST_GCN --device 0 --batch_size 16 --test_batch_size 16

net_suffix = "AE"
train_version = "SGD"
date = 1
# 1. ntu-xsub
# 2. ntu-xview
# 3. kinetics-skeleton

if net_suffix != "":
    net_suffix = "_" + net_suffix

if train_version != "":
    train_version = "_" + train_version

if date == 1:
    dataName_config = "ntu-xsub"
    dataName_workdir = "ntu-xsub"
if date == 2:
    dataName_config = "ntu-xview"
    dataName_workdir = "ntu-xview"    
if date == 3:
    dataName_config = "kinetics-skeleton"
    dataName_workdir = "kinetics_skeleton"

cmd = "python3 main.py recognition -c config/st_gcn/"+ dataName_config +"/train.yaml --model net.st_gcn"+ net_suffix +".Model --work_dir ./work_dir/recognition/"+ dataName_workdir +"/ST_GCN" + net_suffix + train_version
os.system(cmd)

tEnd = time.time()
print('train time: ' + str(tEnd - tStart) + 'sec')
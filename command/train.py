import os
import time

tStart = time.time()

cmd = "python3 main.py recognition -c config/st_gcn/kinetics-skeleton/train.yaml"
# cmd = "python3 main.py recognition -c config/st_gcn/ntu-xsub/train.yaml --device 0 --batch_size 16 --test_batch_size 16"
# cmd = "python3 main.py recognition -c config/st_gcn/ntu-xview/train.yaml"
os.system(cmd)

tEnd = time.time()
print('train time: ' + str(tEnd - tStart) + 'sec')
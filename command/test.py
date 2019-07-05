import os
import time

tStart = time.time()

cmd = "python3 main.py recognition -c config/st_gcn/kinetics-skeleton/test.yaml"
os.system(cmd)

tEnd = time.time()
print('test time: ' + str(tEnd - tStart) + 'sec')
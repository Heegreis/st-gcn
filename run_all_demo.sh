#!/bin/bash

for filename in data/mydata/test_video/0601/nolag/*.avi
do echo $filename
   python3 main.py demo_custom --openpose /openpose/build --video $filename
done
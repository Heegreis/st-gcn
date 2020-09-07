#!/bin/bash

for filename in data/mydata/test_video/sun_video/正式/*.avi
do echo $filename
   python3 main.py demo_custom --openpose /openpose/build --video $filename
done
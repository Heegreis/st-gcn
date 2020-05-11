#!/usr/bin/env python
import os
import sys
import argparse
import json
import shutil
import time

import numpy as np

import cv2

openpose_path = '/openpose/build'
# load openpose python api
if openpose_path is not None:
    sys.path.append('{}/python'.format(openpose_path))
    sys.path.append('{}/build/python'.format(openpose_path))
try:
    from openpose import pyopenpose as op
except ImportError as e:
    print('Can not find Openpose Python API.')
    raise e

# initiate
opWrapper = op.WrapperPython()
params = dict(model_folder='./models', model_pose='COCO')
opWrapper.configure(params)
opWrapper.start()

def videp_process():
    video_path = '00005.MTS'
    video_name = video_path.split('/')[-1].split('.')[0]
    
    label_name_path = './resource/kinetics_skeleton/label_name.txt'
    with open(label_name_path) as f:
        label_name = f.readlines()
        label_name = [line.rstrip() for line in label_name]

    video_capture = cv2.VideoCapture(video_path)

    # start recognition
    start_time = time.time()
    frame_index = 0

    json_data = {
        "label": "balloon blowing",
        "label_index": 12
    }
    openpose_data = []

    while(True):
        frame_index += 1
        skeletons = []

        frame_data = {'frame_index': frame_index}
        # openpose_data.append

        tic = time.time()

        # get image
        ret, orig_image = video_capture.read()
        if orig_image is None:
            break
        source_H, source_W, _ = orig_image.shape
        orig_image = cv2.resize(
            orig_image, (256 * source_W // source_H, 256))
        H, W, _ = orig_image.shape

        # pose estimation
        datum = op.Datum()
        datum.cvInputData = orig_image
        opWrapper.emplaceAndPop([datum])
        multi_pose = datum.poseKeypoints  # (num_person, num_joint, 3)

        # normalization
        multi_pose[:, :, 0] = multi_pose[:, :, 0]/W
        multi_pose[:, :, 1] = multi_pose[:, :, 1]/H
        multi_pose[:, :, 0][multi_pose[:, :, 2] == 0] = 0
        multi_pose[:, :, 1][multi_pose[:, :, 2] == 0] = 0

        if len(multi_pose.shape) != 3:
            continue
        for person in multi_pose:
            score, coordinates = [], []
            skeleton = {}
            for i in range(0, len(person)):
                keypoint = person[i]
                x = round(keypoint[0].item(), 3)
                y = round(keypoint[1].item(), 3)
                coordinates += [x, y]
                score += [round(keypoint[2].item(), 3)]
            skeleton['pose'] = coordinates
            skeleton['score'] = score
            skeletons += [skeleton]
        
        frame_data['skeleton'] = skeletons

        openpose_data.append(frame_data)

    json_data['data'] = openpose_data

    emb_filename = ('emb_json.json')  
    jsObj = json.dumps(json_data)
    with open(emb_filename, "w") as f:
        f.write(jsObj)
        f.close()


if __name__ == "__main__":
    videp_process()
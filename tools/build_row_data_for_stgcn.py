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

def get_video_WH(video_path):
    capture = cv2.VideoCapture(video_path)
    W, H = 0, 0
    while(True):
        # get image
        ret, orig_image = capture.read()
        if not (orig_image is None):
            source_H, source_W, _ = orig_image.shape
            if source_H != 0 and source_W != 0:
                W, H = 256 * source_W // source_H, 256
                break
    capture.release()
    print(W, H)
    return W, H

def getRectFromSkeleton(poseData):
    first = True
    for keypoint in poseData:
        if keypoint[2] != 0:
            if first:
                max_x, min_x = keypoint[0], keypoint[0]
                max_y, min_y = keypoint[1], keypoint[1]
                first = False
            else:
                if keypoint[0] > max_x:
                    max_x = keypoint[0]
                if keypoint[0] < min_x:
                    min_x = keypoint[0]
                if keypoint[1] > max_y:
                    max_y = keypoint[1]
                if keypoint[1] < min_y:
                    min_y = keypoint[1]
    return min_x, min_y, max_x, max_y

def videp_process(video_path, label, output_path):
    video_name = video_path.split('/')[-1].split('.')[0]

    label_name_path = './resource/kinetics_skeleton/label_name.txt'
    with open(label_name_path) as f:
        label_name = f.readlines()
        label_name = [line.rstrip() for line in label_name]

    width, height = get_video_WH(video_path)
    video_capture = cv2.VideoCapture(video_path)
    video_output_path = os.path.join(output_path, 'videos', label, video_name + '.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    print(video_name)
    out = cv2.VideoWriter(video_output_path, fourcc, 30, (width, height))

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

        if len(multi_pose.shape) != 3:
            frame_data['skeleton'] = skeletons
            openpose_data.append(frame_data)
            continue
        # drow skeleton on img
        frameToView = orig_image
        for person in multi_pose:
            min_x, min_y, max_x, max_y = getRectFromSkeleton(person)
            cv2.rectangle(frameToView, (int(min_x), int(min_y)),
                          (int(max_x), int(max_y)), (255, 0, 0), 2)
        out.write(frameToView)

        # normalization
        multi_pose[:, :, 0] = multi_pose[:, :, 0]/W
        multi_pose[:, :, 1] = multi_pose[:, :, 1]/H
        multi_pose[:, :, 0][multi_pose[:, :, 2] == 0] = 0
        multi_pose[:, :, 1][multi_pose[:, :, 2] == 0] = 0

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
    json_file_path = os.path.join(output_path, 'poses', label, video_name + '.json')
    jsObj = json.dumps(json_data)
    with open(json_file_path, "w") as f:
        f.write(jsObj)
        f.close()

    video_capture.release()
    out.release()

def mkdirs(out_videos_path, labels):
    videos_path = os.path.join(out_videos_path, 'videos')
    poses_path = os.path.join(out_videos_path, 'poses')

    for label in labels:
        video_label_path = os.path.join(videos_path, label)
        pose_label_path= os.path.join(poses_path, label)
        if os.path.isdir(video_label_path) == False:
            os.makedirs(video_label_path)
        if os.path.isdir(pose_label_path) == False:
            os.makedirs(pose_label_path)

def loop_dir():
    # dataset_path = 'dataset/row_video_for_train'
    dataset_path = 'dataset/test_row_video_for_train'
    output_path = 'dataset/test/custom_skeleten_data'

    dirs = os.listdir(dataset_path) # label name
    mkdirs(output_path, dirs)
    # 输出所有文件和文件夹
    for label in dirs:
        # print(label)
        class_path = os.path.join(dataset_path, label)
        files = os.listdir(class_path)
        for file_name in files:
            video_path = os.path.join(class_path, file_name)
            videp_process(video_path, label, output_path)

if __name__ == "__main__":
    loop_dir()
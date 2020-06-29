import cv2
import numpy as np
import torch

from lightOpenPose.models.with_mobilenet import PoseEstimationWithMobileNet
from lightOpenPose.modules.keypoints import extract_keypoints, group_keypoints
from lightOpenPose.modules.load_state import load_state
from lightOpenPose.modules.pose import Pose
from lightOpenPose.val import normalize, pad_width

import time

def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad

def convert_to_openpose_format(pose_entries, all_keypoints):
    keypoints = np.zeros((len(pose_entries), 18, 3), dtype=np.float32)
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        position_id = -1
        for keypoint_id in pose_entries[n][:-2]:
            position_id += 1

            cx, cy, score = 0, 0, 0  # keypoint not found
            if keypoint_id != -1:
                cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
            keypoints[n][position_id][0] = cx * 2
            keypoints[n][position_id][1] = cy * 2
            keypoints[n][position_id][2] = score

    return keypoints

class LightOpenPose():
    def __init__(self, height_size):
        checkpoint_path = 'lightOpenPose/checkpoint_iter_370000.pth'
        self.cpu = False

        self.net = PoseEstimationWithMobileNet()
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        load_state(self.net, checkpoint)
        
        self.net = self.net.eval()
        if not self.cpu:
            self.net = self.net.cuda()

        self.height_size = height_size
        self.stride = 8
        self.upsample_ratio = 4
        self.num_keypoints = Pose.num_kpts

    def getPose(self, img):
        time_light_start = time.time()
        heatmaps, pafs, scale, pad = infer_fast(self.net, img, self.height_size, self.stride, self.upsample_ratio, self.cpu)
        # print('time_light_start: ' + str(time.time() - time_light_start))

        time_light_for = time.time()
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(self.num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)
        # print('time_light_for: ' + str(time.time() - time_light_for))
        
        time_light_group = time.time()
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=False)
        # print('time_light_group: ' + str(time.time() - time_light_group))

        time_light_convert = time.time()
        keypoints = convert_to_openpose_format(pose_entries, all_keypoints)
        # print('time_light_convert: ' + str(time.time() - time_light_convert))

        return keypoints
    

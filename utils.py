import torch
import cv2
import os
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler

def one_hot_encode(index, num):
    vector = [0 for _ in range(num)]
    vector[index] = 1
    return torch.Tensor(vector)

def extract_frames(video_path, save_path, fps=5):
    video_name = video_path.split('\\')[-1].split('.')[0]
    extracted_path = os.path.join(save_path, video_name)

    if not os.path.isdir(extracted_path):
        os.mkdir(extracted_path)
    else:
        raise IOError(f'Folder {extracted_path} already exists')

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = video_fps // fps 

    if not cap.isOpened():
        raise IOError(f'Cannot read {video_path}. The file is an invalid video  or does not exist.')

    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret:
            count += 1
            if count % frame_interval == 0:
                cv2.imwrite(os.path.join(extracted_path, f'{video_name}{count:003}.jpg'), frame)
        else:
            break;
    return extracted_path





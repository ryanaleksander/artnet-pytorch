import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import evaluation
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data  import DataLoader
from configparser import ConfigParser
import argparse

import utils
from video_dataset import VideoFramesDataset
from artnet import ARTNet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to config file', required=True)
    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)

    test_loader, class_list = load_data(config['Test Data'])
    accuracy = test(config['Test'], test_loader, class_list)
    print(accuracy)

def load_data(params):
    print('Loading data...')
    transform = transforms.Compose([
        transforms.Resize((params.getint('width'), params.getint('height'))),
        transforms.RandomCrop((params.getint('crop'), params.getint('crop'))),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_set = VideoFramesDataset(params['path'], transform=transform)
    class_list = test_set.cls_lst

    print('Done loading data')
    return DataLoader(test_set, batch_size=1), class_list

def test(params, test_loader, class_list):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    artnet = ARTNet(num_classes=len(class_list))
    artnet.load_state_dict(torch.load(params['model']))
    artnet = artnet.to(device)

    testing_progress = tqdm(enumerate(test_loader))
    testing_result = []
    ground_truths = []
    batch_size = params.getint('batch_size')
    frame_num = params.getint('frame_num')

    for batch_index, (frames, label) in testing_progress:
        testing_progress.set_description('Batch no. %i: ' % batch_index)

        # Ensure that all samples have the equal amount of frames
        leftover = frames.size()[1] % frame_num
        if leftover != 0:
            frames = torch.cat((frames, frames[:,-frame_num+leftover:,:,:,:]), dim=1)

        # Split all frames into frame groups
        frames = torch.split(frames, frame_num, dim=1)
        frames = torch.cat(frames)
        predictions = torch.zeros((1, len(class_list)))
        ground_truths.append(label)
        for i in range(0, frames.size()[0], batch_size):
            input = frames[i:i+batch_size]
            input = input.to(device)
            output = artnet(input)
            output = F.softmax(output, dim=1)
            output = output.sum(dim=0)
            predictions += output
        testing_result.append(predictions.argmax().item())

    testing_result = torch.Tensor(testing_result)
    ground_truths = torch.Tensor(ground_truths)
    accuracy = torch.eq(testing_result, ground_truths).sum() / len(ground_truths)


if __name__ == '__main__':
    main()

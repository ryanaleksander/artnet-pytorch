import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
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

    test_loader = load_data(config['Test Data'])
    print(len(test_loader))
    confusion_matrix = test(config['Train'], test_loader)

def load_data(params):
    """Load data for training"""

    print('Loading data...')
    transform = transforms.Compose([
        transforms.Resize((params.getint('width'), params.getint('height'))),
        transforms.RandomCrop((params.getint('crop'), params.getint('crop'))),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    
    test_set = VideoFramesDataset(params['path'], transform=transform)
    print('Done loading data')
    return DataLoader(test_set, batch_size=params.getint('batch_size'))

def test(params, test_loader):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    artnet = ARTNet(num_classes=params.getint('num_classes'))
    artnet = artnet.to(device)

    testing_progress = tqdm(enumerate(test_loader))

    results = []
    ground_truths = []
    for batch_index, (frames, label) in testing_progress:
        testing_progress.set_description('Batch no. %i: ' % batch_index)

        output = artnet(frames)
        results.extends(list(output.argmax(dim=1)))
        ground_truths.extends(label)
    
    cls_lst = params['cls_lst'].split(',')
    
    return calculate_confustion_matrix(results, ground_truths, cls_lst)

def calculate_confustion_matrix(results, ground_truths, cls_lst):
    matrix = {
        'TP': {k: 0 for k in cls_lst},
        'FP': {k: 0 for k in cls_lst},
        'TN': {k: 0 for k in cls_lst},
        'FN': {k: 0 for k in cls_lst}
    }

    for i in range(len(labels)):
        if labels[i] == ground_truths[i]:
            matrix['TP'][cls_lst[labels[i]]] += 1
            for label in cls_lst:
                if label != cls_lst[labels[i]]:
                    matrix['TN'][label] += 1
        else:
            matrix['FN'][cls_lst[ground_truths[i]]] += 1
            matrix['FP'][cls_lst[labels[i]]] += 1
            for label in cls_lst:
                if label != cls_lst[labels[i]] and label != cls_lst[ground_truths[i]]:
                    matrix['TN'][label] += 1
    return matrix

if __name__ == '__main__':
    main()




        

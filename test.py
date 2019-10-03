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
    train_losses, val_losses = test(config['Train'], test_loader)
    #save_result(train_losses, val_losses, config['Train Result'])

def load_data(params):
    """Load data for training"""

    print('Loading data...')
    transform = transforms.Compose([
        transforms.Resize((params.getint('width'), params.getint('height'))),
        transforms.RandomCrop((params.getint('crop'), params.getint('crop'))),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

def calculate_confustion_matrix(results, ground_truths, cls_lst):
    matrix = {
        'TP': {k: 0 for k in cls_lst},
        'FP': {k: 0 for k in cls_lst},
        'TN': {k: 0 for k in cls_lst},
        'FN': {k: 0 for k in cls_lst}
    }

    labels = results.argmax(dim=1)

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


def save_result(train_losses, val_losses, params):
    """Saving result in term of training loss and validation loss"""

    # Save chart
    data = { 'epoch': range(1, len(train_losses) + 1), 'train': train_losses, 'val': val_losses}
    plt.plot('epoch', 'train', data=data, label='Training loss', color='blue' )
    plt.plot('epoch', 'val', data=data, label='Validation loss', color='red' )
    plt.legend()
    plt.savefig(os.path.join(params['path'], 'result.png'))

    # Save log
    file_path = os.path.join(os.path.join(params['path'], result.txt))
    with open(file_path, 'w') as f:
        for i in range(len(train_losses)):
            f.write('Epoch %i: training loss - %0.4f, validation loss - %0.4f\n' % (i + 1, train_losses[i], val_losses[i]))

    
if __name__ == '__main__':
    main()




        

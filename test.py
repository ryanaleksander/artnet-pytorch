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
    artnet.load_state_dict(torch.load(params['model'], map_location=torch.device('cpu')))
    artnet = artnet.to(device)

    eval_scheme = evaluation.ConsecutiveSequencesDetectionEvaluation(pos_class=params['positive'], num_sequence=params.getint('num_sequence'))

    testing_progress = tqdm(enumerate(test_loader))
    testing_result = []
    ground_truths = []
    batch_size = params.getint('batch_size')

    for batch_index, (frames, label) in testing_progress:
        testing_progress.set_description('Batch no. %i: ' % batch_index)
        frame_num = params.getint('frame_num')
        predictions = []
        frames = frames.to(device)
        frames = torch.split(frames, frame_num, dim=1)
        frames = torch.cat(frames)
        ground_truths.append(params['positive'] == class_list[label])
        for i in range(0, frames.size()[0], batch_size):
            input = frames[i:i+batch_size]
            output = artnet(input)
            output = F.softmax(output, dim=1)
            result = output.argmax(dim=1)
            predictions.append([class_list[res] for res in result])
        testing_result.append(eval_scheme.eval(predictions))
    #correct_predictions = [testing_result[i] == ground_truths[i] for i in range(len(testing_result))]
    testing_result = [int(res) for res in testing_result]
    ground_truths = [int(gt) for gt in ground_truths]

    return calculate_confusion_matrix(testing_result, ground_truths, class_list)

def calculate_confusion_matrix(result, ground_truths, cls_lst):
    matrix = {
        'TP': {k: 0 for k in cls_lst},
        'FP': {k: 0 for k in cls_lst},
        'TN': {k: 0 for k in cls_lst},
        'FN': {k: 0 for k in cls_lst}
    }

    for i in range(len(result)):
        if result[i] == ground_truths[i]:
            matrix['TP'][cls_lst[result[i]]] += 1
            for label in cls_lst:
                if label != cls_lst[result[i]]:
                    matrix['TN'][label] += 1
        else:
            matrix['FN'][cls_lst[ground_truths[i]]] += 1
            matrix['FP'][cls_lst[result[i]]] += 1
            for label in cls_lst:
                if label != cls_lst[result[i]] and label != cls_lst[ground_truths[i]]:
                    matrix['TN'][label] += 1
    return matrix

if __name__ == '__main__':
    main()




        

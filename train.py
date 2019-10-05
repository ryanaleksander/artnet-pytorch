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

    train_loader, validation_loader = load_data(config['Train Data'])
    print(len(validation_loader))
    train_losses, val_losses = train(config['Train'], train_loader, validation_loader)
    save_result(train_losses, val_losses, config['Train Result'])

def load_data(params):
    """Load data for training"""

    print('Loading data...')
    transform = transforms.Compose([
        transforms.Resize((params.getint('width'), params.getint('height'))),
        transforms.RandomCrop((params.getint('crop'), params.getint('crop'))),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_set = VideoFramesDataset(params['path'], transform=transform)
    dataset_size = len(train_set)
    indices = list(range(dataset_size))
    split = int(np.floor(params.getfloat('val_split') * dataset_size))

    # Shuffle dataset
    if params.getboolean('shuffle'):
        np.random.seed(42)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(train_set, batch_size=params.getint('batch_size'), sampler=train_sampler)
    validation_loader = DataLoader(train_set, batch_size=params.getint('batch_size'), sampler=valid_sampler)
    print('Done loading data')

    return train_loader, validation_loader

def train(params, train_loader, validation_loader):
    artnet = ARTNet(num_classes=params.getint('num_classes'))
    device = 'cuda'

    # Load pretrained model
    if 'pretrained' in params and params['pretrained'] is not None:
        artnet.load_state_dict(torch.load(params['pretrained']))

    #  Determine training devices
    if params.getboolean('cuda'):
        devices = ['cuda:' + id for id in params['gpus'].split(',')]
        if len(devices) > 1:
            artnet = nn.DataParallel(artnet, device_ids=devices)
    else:
        device = 'cpu'
    artnet = artnet.to(device)

    optimizer = optim.SGD(artnet.parameters(), lr=params.getfloat('lr'), momentum=params.getfloat('momentum'))
    criterion = nn.CrossEntropyLoss()

    # Learning rate decay config
    lr_steps = [int(step) for step in params.get('lr_steps').split(',')]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=0.1)

    for epoch in range(params.getint('num_epochs')):
        print('Starting epoch %i:' % (epoch + 1))
        print('*********Training*********')
        artnet.train()
        training_loss = 0
        training_losses = []
        training_progress = tqdm(enumerate(train_loader))
        correct = 0
        for batch_index, (frames, label) in training_progress:
            training_progress.set_description('Batch no. %i: ' % batch_index)
            frames, label = frames.to(device), label.to(device)

            optimizer.zero_grad()
            output = artnet(frames)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

            # Calculating accuracy
            _, prediction = output.max(dim=1)
            prediction, label = prediction.to('cpu'), label.to('cpu')
            correct += prediction.eq(torch.LongTensor(label)).sum()

        else:
            avg_loss = training_loss / len(train_loader)
            accuracy = correct / len(train_loader) * train_loader.batch_size
            training_losses.append(avg_loss)
            print(f'Training loss: {avg_loss}')
            print(f'Training accuracy: {accuracy:0.2f}')

        print('*********Validating*********')
        artnet.eval()
        validating_loss = 0
        validating_losses = []
        validating_progress = tqdm(enumerate(validation_loader))
        with torch.no_grad():
            for batch_index, (frames, label) in validating_progress:
                validating_progress.set_description('Batch no. %i: ' % batch_index)
                frames, label = frames.to(device), label.to(device)

                output = artnet(frames)
                loss = criterion(output, label)
                validating_loss += loss.item()

                # Calculating accuracy
                _, prediction = output.max(dim=1)
                prediction, label = prediction.to('cpu'), label.to('cpu')
                correct += prediction.eq(torch.LongTensor(label)).sum()
            else:
                avg_loss = validating_loss / len(validation_loader)
                accuracy = correct / len(train_loader) * validation_loader.batch_size
                validating_losses.append(avg_loss)
                print(f'Validation loss: {avg_loss}')
                print(f'Validation accuracy: {accuracy:0.2f}')
        print('=============================================')
        print('Epoch %i complete' % (epoch + 1))

        if (epoch + 1) % params.getint('ckpt') == 0:
            print('Saving checkpoint...' )
            torch.save(artnet.state_dict(), os.path.join(params['ckpt_path'], 'artnet_%i.pth' % (epoch + 1)))

        # Update LR
        scheduler.step()
    print('Training complete, saving final model....')
    torch.save(artnet.state_dict(), os.path.join(params['ckpt_path'], 'artnet_final.pth'))
    return training_losses, validating_losses

def save_result(train_losses, val_losses, params):
    """Saving result in term of training loss and validation loss"""

    # Save chart
    data = { 'epoch': range(1, len(train_losses) + 1), 'train': train_losses, 'val': val_losses}
    plt.plot('epoch', 'train', data=data, label='Training loss', color='blue' )
    plt.plot('epoch', 'val', data=data, label='Validation loss', color='red' )
    plt.legend()
    plt.savefig(os.path.join(params['path'], 'result.png'))

    # Save log
    file_path = os.path.join(os.path.join(params['path'], 'result.txt'))
    with open(file_path, 'w') as f:
        for i in range(len(train_losses)):
            f.write('Epoch %i: training loss - %0.4f, validation loss - %0.4f\n' % (i + 1, train_losses[i], val_losses[i]))

    
if __name__ == '__main__':
    main()




        

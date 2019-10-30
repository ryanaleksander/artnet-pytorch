# ARTNet Pytorch
## Introduction
Pytorch implementation of ARTNet - Appearance and Relation Network for Video Classification

Paper: http://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Appearance-and-Relation_Networks_for_CVPR_2018_paper.html

Original Caffe implementation: https://github.com/wanglimin/ARTNet

## Requirements
The code is written using the following environment. There isn't a strict version requirement, but deviate from the listed versions at your own risk

* python: 3.7.3
* pytorch: 1.2.0
* torchvision: 0.4.0
* matplotlib: 3.1.0
* numpy: 1.16.4
* tqdm: 4.32.1

## Training
### Preparing Data
The dataset folder should have the following structure
```
train  
│
└───category1
│   │   
│   │
│   └───video1
│       │   frame001.png
│       │   frame002.png
│       │   ...
│   
└───category2
    │   
    │   ... 
```

Video frames have to be extracted prior to the training process. This repository, as of now, does not provide means to extract video frames.
### Configuration
1. Make a copy of `config.ini`
2. Edit the configurations as you see fit
### Run
`python train.py --config [config file path]`  

## Testing
[TODO]

## FAQ
**Q. Have you tested the code on any standard datasets?**

A. As of now, no. I've been having trouble downloading the Kinetic dataset in full, so I've only been able to test the code on a few categories. I'll provide the full testing statistics as soon as possible.
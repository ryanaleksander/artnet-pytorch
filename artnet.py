import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SquarePool(nn.Module):
    """Square Pooling Layer"""
    def __init__(self):
        super(SquarePool, self).__init__()

    def forward(self, x):
        return x**2

class CrossChannelPool(nn.Module):
    """Cross Channel Pooling layer using 1x1x1 kernel to aggregrate weights across channels"""
    def __init__(self, in_channels, out_channels):
        super(CrossChannelPool, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1,1,1), stride=(1,1,1))
        self.conv.weight.fill_(0.5)

    def forward(self, x):
        return self.conv(x)

class SMART(nn.Module):
    """SMART block with two branchs: Relation and Appearance"""
    def __init__(self, in_channels, out_channels, kernel, stride=(1,1,1), padding=(0, 0, 0)):
        super(SMART, self).__init__()
        self.relation = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm3d(64),
            SquarePool(),
            CrossChannelPool(64, 32),
            nn.BatchNorm3d(32),
        )

        # Convolution params for appearance branch to make Conv3D operates like Conv2D
        app_kernel = list(kernel[:2])
        app_kernel.append(1)
        app_padding = list(padding[:2])
        app_padding.append(0)

        self.appearance = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=app_kernel, stride=stride, padding=app_padding),
            nn.BatchNorm3d(64),
        )

        self.reduction = nn.Conv3d(64 + 32, out_channels, kernel_size=(1,1,1), bias=False)

    def forward(self, x):
        x_rel = self.relation(x)
        x_app = self.appearance(x)
        out = torch.cat((x_rel, x_app), dim=1)
        out = F.relu(out)
        out = self.reduction(out)
        return out

class C3D_SMART(nn.Module):
    """ResNet block composing of a C3D and a SMART block"""
    def __init__(self, in_channels, out_channels, kernel, stride=(1,1,1), padding=(0,0,0)):
        super(C3D_SMART, self).__init__()

        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel=kernel, stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.smart = SMART(out_channels, out_channels, kernel=kernel, stride=(1,1,1), padding=padding)

    def forward(self, x):
        out = self.conv3d(x)
        out = F.relu(self.bn(out))
        out = self.smart(out)
        return out

class ARTNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ARTNet, self).__init__()

        self.num_classes = num_classes

        # Conv1
        self.conv1 = SMART(3, 64, kernel=(7,7,3), stride=(2,2,2), padding=(3, 3, 1))
        self.conv1_bn = nn.BatchNorm3d(64)

        # Conv2
        self.conv2_1 = C3D_SMART(64, 64, kernel=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.resnet2a_bn = nn.BatchNorm3d(64)
        self.conv2_2 = C3D_SMART(64, 64, kernel=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.resnet2b_bn = nn.BatchNorm3d(64)

        # Conv3
        self.resnet3a_down = nn.Conv3d(64, 128, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
        self.conv3_1 = C3D_SMART(64, 128, kernel=(3,3,3), stride=(2,2,2), padding=(1,1,1))
        self.resnet3a_bn = nn.BatchNorm3d(128)
        self.conv3_2 = C3D_SMART(128, 128, kernel=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.resnet3b_bn = nn.BatchNorm3d(128)

        # Conv4
        self.resnet4a_down = nn.Conv3d(128, 256, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
        self.conv4_1 = C3D_SMART(128, 256, kernel=(3,3,3), stride=(2,2,2), padding=(1,1,1))
        self.resnet4a_bn = nn.BatchNorm3d(256)
        self.conv4_2 = C3D_SMART(256, 256, kernel=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.resnet4b_bn = nn.BatchNorm3d(256)

        # Conv5
        self.resnet5a_down = nn.Conv3d(256, 512, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
        self.conv5_1 = nn.Sequential(
            nn.Conv3d(256, 512, kernel=(3,3,3), stride=(2,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel=(3,3,3), stride=(2,2,2), padding=(1,1,1)),
        )
        self.resnet5a_bn = nn.BatchNorm3d(512)
        self.conv5_2 = nn.Sequential(
            nn.Conv3d(512, 512, kernel=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
        )
        self.resnet5b_bn = nn.BatchNorm3d(512)

        # Global Pool
        self.avgpool = nn.AvgPool3d(kernel_size=(7,7,1), stride=(1,1,1))
        self.dropout = nn.Dropout3d(p=0.2)

        # Fully Connected
        self.fc8 = nn.Linear(512, self.num_classes)

    def resnet(self, input, identity, conv, bn, downsample=None, bn_relu=False):
        input = conv(input)

        if downsample is not None:
            identity = downsample(identity)

        input += identity
        identity = input
        input = bn(input)
        input = F.relu(input)

        if bn_relu:
            identity = input

        return input, identity

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[2], x.shape[3], x.shape[4], x.shape[1])

        # Conv1
        x = self.conv1(x)
        identity = x
        x = self.conv1_bn(x)
        x = F.relu(x)

        # Conv2
        x, identity = self.resnet(x, identity, self.conv2_1, self.resnet2a_bn)
        x, identity = self.resnet(x, identity, self.conv2_2, self.resnet2b_bn, bn_relu=True)

        # Conv3
        x, identity = self.resnet(x, identity, self.conv3_1, self.resnet3a_bn, downsample=self.resnet3a_down)
        x, identity = self.resnet(x, identity, self.conv3_2, self.resnet3b_bn, bn_relu=True)

        # Conv4
        x, identity = self.resnet(x, identity, self.conv4_1, self.resnet4a_bn, downsample=self.resnet4a_down)
        x, identity = self.resnet(x, identity, self.conv4_2, self.resnet4b_bn, bn_relu=True)

        # Conv5
        x, identity = self.resnet(x, identity, self.conv5_1, self.resnet5a_bn, downsample=self.resnet5a_down)
        x, identity = self.resnet(x, identity, self.conv5_2, self.resnet5b_bn)

        x = self.avgpool(x)
        x = x.view(-1, 512)
        x = self.fc8(x)

        return x
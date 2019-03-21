#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(1, 20,
                               kernel_size=5,
                               stride=1,bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2,
                                  stride=2)
        self.conv2 = nn.Conv2d(20, 50,
                               kernel_size=5,
                               stride=1,bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2,
                                  stride=2)
        self.fc1 = nn.Linear(50*4*4, 500,bias=False)
        self.fc2 = nn.Linear(500, 10,bias=False)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        output = self.fc2(x)
        return output

class DIYLeNet(nn.Module):
    def __init__(self,conv1_num,conv2_num):
        super(DIYLeNet,self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_num,
                               kernel_size=5,
                               stride=1,bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2,
                                  stride=2)
        self.conv2 = nn.Conv2d(conv1_num, conv2_num,
                               kernel_size=5,
                               stride=1,bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2,
                                  stride=2)
        self.fc1 = nn.Linear(conv2_num*4*4, 500,bias=False)
        self.fc2 = nn.Linear(500, 10,bias=False)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        output = self.fc2(x)
        return output
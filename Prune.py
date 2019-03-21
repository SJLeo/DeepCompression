#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import os
import numpy as np
from torch.nn import Parameter
from LeNet import LeNet,DIYLeNet

zero_threshode = 0.001

if __name__ == '__main__':
    with open(os.path.join('./model','SSL_Model.pt'),'rb') as f:
        model = torch.load(f,map_location='cpu')

    conv1_weight = model.conv1.weight.data
    conv2_weight = model.conv2.weight.data
    fc1_weight = model.fc1.weight.data
    fc2_weight = model.fc2.weight.data

    cond = torch.abs(model.conv1.weight) < zero_threshode
    model.conv1.weight.data[cond] = 0
    cond = torch.abs(model.conv2.weight) < zero_threshode
    model.conv2.weight.data[cond] = 0
    # cond = torch.abs(model.fc1.weight) < zero_threshode
    # model.fc1.weight.data[cond] = 0
    # cond = torch.abs(model.fc2.weight) < zero_threshode
    # model.fc2.weight.data[cond] = 0

    conv2_num = 0
    for i in range(model.conv2.weight.size(0)):
        if torch.sum(model.conv2.weight[i]) != 0:
            conv2_num += 1

    conv1_num = 0
    for i in range(model.conv2.weight.size(1)):
        if torch.sum(model.conv2.weight[:,i,:,:]) != 0:
            conv1_num += 1

    SSLmodel = DIYLeNet(conv1_num,conv2_num)

    conv1_index = 0
    newconv1_weight = torch.FloatTensor(SSLmodel.conv1.weight.shape)
    for i in range(conv2_weight.size(1)):
        if torch.sum(conv2_weight[:,i,:,:]) != 0:
            newconv1_weight[conv1_index] = conv1_weight[i]
            conv1_index += 1

    conv2_index = 0
    newconv2_weight = torch.FloatTensor(SSLmodel.conv2.weight.shape)
    for i in range(conv2_weight.size(0)):
        if torch.sum(conv2_weight[i]) != 0:
            index_j = 0
            for j in range(conv2_weight.size(1)):
                if torch.sum(conv2_weight[i][j]) != 0:
                    newconv2_weight[conv2_index][index_j] = conv2_weight[i][j]
                    index_j +=1
            conv2_index += 1

    for i in range(conv2_weight.size(0)-1,-1,-1):
        if torch.sum(conv2_weight[i]) == 0:
            fc1_weight = np.delete(fc1_weight,np.arange(i*4*4,(i+1)*4*4),axis=1)

    SSLmodel.conv1.weight = Parameter(newconv1_weight)
    SSLmodel.conv2.weight = Parameter(newconv2_weight)
    SSLmodel.fc1.weight = Parameter(fc1_weight)
    SSLmodel.fc2.weight = Parameter(fc2_weight)

    with open(os.path.join('./model','Prune_Model.pt'),'wb') as f:
        torch.save(SSLmodel,f)

    print('Prune done!')


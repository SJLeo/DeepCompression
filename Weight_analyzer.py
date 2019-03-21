#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import os
from LeNet import LeNet

zero_threshode= 0.001

if __name__ == '__main__':

    with open(os.path.join('./model','SSL_Model.pt'),'rb') as f:
        model = torch.load(f,map_location='cpu')
    cond = torch.abs(model.conv1.weight) < zero_threshode
    model.conv1.weight.data[cond] = 0
    cond = torch.abs(model.conv2.weight) < zero_threshode
    model.conv2.weight.data[cond] = 0
    with open(os.path.join('./model','SSL_Model.pt'),'wb') as f:
        torch.save(model,f)

    print('------------------------------------------------------')
    print('Conv1 Filter\'s weight is all zero:')
    zero_num = 0
    for i in range(model.conv1.weight.size(0)):
        if torch.sum(model.conv1.weight[i]) == 0:
            print(i+1,end=' ')
            zero_num += 1
    print()
    print('Conv1 Filter\'s sparse Rate:%d / %d' %(zero_num, model.conv1.weight.size(0)))

    print('------------------------------------------------------')
    print('Conv1 Channel\'s weight is all zero:')
    zero_num = 0
    for i in range(model.conv1.weight.size(1)):
        if torch.sum(model.conv1.weight[:,i,:,:]) == 0:
            print(i + 1,end=' ')
            zero_num +=1
    print()
    print('Conv1 Channel\'s sparse Rate:%d / %d' % (zero_num, model.conv1.weight.size(1)))

    print('------------------------------------------------------')
    print('Conv2 Filter\'s weight is all zero:')
    zero_num = 0
    for i in range(model.conv2.weight.size(0)):
        if torch.sum(model.conv2.weight[i]) == 0:
            print(i + 1, end=' ')
            zero_num += 1
    print()
    print('Conv2 Filter\'s sparse Rate:%d / %d' % (zero_num, model.conv2.weight.size(0)))

    print('------------------------------------------------------')
    print('Conv2 Channel\'s weight is all zero:')
    zero_num = 0
    for i in range(model.conv2.weight.size(1)):
        if torch.sum(model.conv2.weight[:,i,:,:]) == 0:
            print(i + 1,end=' ')
            zero_num +=1
    print()
    print('Conv2 Channel\'s sparse Rate:%d / %d' % (zero_num,model.conv2.weight.size(1)))

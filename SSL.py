#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import os
import argparse
from LeNet import LeNet

parser = argparse.ArgumentParser()
parser.add_argument("--type",type= str,default='SSL')
parser.add_argument("--iter",type=int, default=10)
args = parser.parse_args()

group_weight_decay = 1e-4
learn_rate = 0.001
weight_decay = 5e-4
best_val_loss = None
loss_func = nn.CrossEntropyLoss()

def add_filter_wise_grouplasso(weight):
    wise_square = weight.pow(2)
    total_regular = 0
    for n in range(weight.size(0)):
        total_regular += torch.sqrt(torch.sum(wise_square[n,:,:,:]))
    return total_regular

def add_channel_wise_grouplasso(weight):
    wise_square = weight.pow(2)
    total_regular = 0
    for c in range(weight.size(1)):
        total_regular += torch.sqrt(torch.sum(wise_square[:, c, :, :]))
    return total_regular

def add_l1reg(weight):
    absweight = torch.abs(weight)
    return sum(absweight)

def train(model,  train_loader, optimizer, epoch, type='SSL'):
    model.train()
    for batch, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = loss_func(output, target)
        regular_loss = 0
        l1_reg = 0.
        if type == 'SSL':
            regular_loss = group_weight_decay * add_filter_wise_grouplasso(model.conv1.weight) + \
                           group_weight_decay * add_filter_wise_grouplasso(model.conv2.weight)
            regular_loss += group_weight_decay * add_channel_wise_grouplasso((model.conv1.weight)) + \
                            group_weight_decay * add_channel_wise_grouplasso(model.conv2.weight)
            for param in model.parameters():
                l1_reg += torch.norm(param,1)
            total_loss = loss + regular_loss + weight_decay * l1_reg
        else:
            total_loss = loss
        optimizer.zero_grad()
        total_loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(),10)
        optimizer.step()

        if batch % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tregular: {:.6f}'.format(
                epoch, batch * len(data), len(train_loader.dataset),
                       100. * batch / len(train_loader), loss.item(), regular_loss))

def test(model, test_loader,type = 'SSL'):
    model.eval()
    global best_val_loss
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += loss_func(output,target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss:{:4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))
    if not best_val_loss or test_loss < best_val_loss:
        modelPath = os.path.join('./model',type+'_Model.pt')
        with open(modelPath, 'wb') as f:
            torch.save(model, f)
        best_val_loss = test_loss

if __name__ == '__main__':
    model = LeNet()
    # print(model)

    trainLoader = Data.DataLoader(dataset=torchvision.datasets.MNIST(
        root='./',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    ),batch_size=64,shuffle=True)
    testLoader = Data.DataLoader(dataset=torchvision.datasets.MNIST(
        root='./',
        train=False,
        transform=torchvision.transforms.ToTensor()
    ),batch_size=100, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9)

    for epoch in range(1,args.iter+1):
        # if epoch % 10 == 0:
        #     learn_rate /= 10
        # optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9)
        # print('Learn_Rate:%.4f' % learn_rate)
        train(model, trainLoader, optimizer, epoch,type=args.type)
        test(model, testLoader,type=args.type)
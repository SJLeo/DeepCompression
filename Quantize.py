#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import pickle
import time
import scipy.cluster.vq as scv
import numpy as np
from torch.nn import Parameter
import os
from LeNet import LeNet,DIYLeNet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--conv1_num",type= int)
parser.add_argument("--conv2_num",type=int)
args = parser.parse_args()

loss_func = nn.CrossEntropyLoss()
best_val_loss = None

#获得各层的量化码表
def kmeasModel(model, layers, num_c=16, initials = None):
    #model:模型
    #layers:需要量化的层
    #num_c:各层的量化级别
    #initals:初始聚类中心方式
    codebook = {} #量化码表
    num_c = [num_c] * len(layers)
    state_dict = model.state_dict()

    print("==============Perform K-means=============")
    for index,layer in enumerate(layers):
        print('Eval layer:'+layer)
        W = state_dict[layer+'.weight'].data.flatten()
        W = W[np.where(W != 0)]


        if initials is None:
            min_W = torch.min(W)
            max_W = torch.max(W)
            initial_uni = np.linspace(min_W, max_W, num_c[index])
            codebook[layer], _ = scv.kmeans(W, initial_uni)

    return codebook

#得到网络的量化权重值
def quantizeModel(model, codebook):
    layers = codebook.keys()
    codes_W = {}
    state_dict = model.state_dict()

    print("================Perform quantization==============")
    for layer in enumerate(layers):
        print('Quantize layer',layer)
        W = state_dict[layer+'.weight'].data
        codes, _ = scv.vq(W.flatten(), codebook[layer])
        codes = np.reshape(codes, W.shape)
        codes_W[layer] = np.array(codes,dtype =np.uint32)

        W_q = np.reshape(codebook[layer][codes], W.shape)
        np.copyto(state_dict[layer+'.weight'],Parameter(W_q))

    return codes_W

#获取每个量化中心在权重矩阵中的位置
def quantizeModelwithDict(model,layers,codebook,timing=False):
    start_time = time.time()
    codeDict = {} #记录各个量化中心在权重矩阵中所处位置
    maskCode = {} #各层量化结果
    state_dict = model.state_dict()
    for layer in layers:
        print("Quantize layer:",layer)
        W = state_dict[layer+'.weight'].data
        codes, _ = scv.vq(W.flatten(), codebook[layer])
        W_q = np.reshape(codebook[layer][codes], W.shape)
        state_dict[layer+'.weight'] = torch.FloatTensor(W_q)

        maskCode[layer] = np.reshape(codes,W.shape)
        a = maskCode[layer].flatten()
        b = range(len(a))

        codeDict[layer] = {}
        for i in range(len(a)):
            codeDict[layer].setdefault(a[i],[]).append(b[i])

    model.load_state_dict(state_dict)

    if timing:
        print('Update codebook time:%f' %(time.time()- start_time))

    return codeDict,maskCode

# def static_vars(**kwargs):
#     def decorate(func):
#         for k in kwargs:
#             setattr(func, k, kwargs[k])
#         return func
#     return decorate
#
# @static_vars(step_cache={}, step_cache2={}, count=0)
def updateCodebook(model , codebook, codeDict, maskCode,layers,grads):

    extra_lr = 0.001
    #权重码表更新
    newstate_dict = model.state_dict()
    for layer in layers:
        diff = grads[layer].data.numpy().flatten()#误差梯度
        codeBookSize = len(codebook[layer])
        dx = np.zeros((codeBookSize))
        for code in range(codeBookSize):
            indexes = codeDict[layer][code] #codeDict保存属于该编码的权重序号

            diff_ave = np.sum(diff[indexes])

            dx[code] = -extra_lr * diff_ave

        codebook[layer] += dx

        newstate_dict[layer+'.weight'] = torch.FloatTensor(codebook[layer][maskCode[layer]])

    model.load_state_dict(newstate_dict)

def saveQuantizeModel(codebook, maskcode, filename, total_layers):
    # 编码
    quantizeModel = {}
    for layer in total_layers:
        quantizeModel[layer+'_codebook'] = np.float32(codebook[layer])
        quantizeModel[layer + '_maskcode'] = np.uint8(maskcode[layer])
        # quantizeModel[layer + '_bias'] = np.float32(state_dict[layer+'.bias'].data)

    pickle.dump(quantizeModel,open(os.path.join('./bin',filename+'.bin'),'wb'))

#恢复权重值
def recover_all(model,layers, filename='Quantize'):
    try:
        with open(os.path.join('./bin',filename+'.bin'),'rb+') as f:
            quantizeModel = pickle.load(f)
    except EOFError:
        return None

    codebook ={}
    maskCode ={}
    biases = {}

    for layer in layers:
        codebook[layer] = quantizeModel[layer+'_codebook']
        maskCode[layer] = quantizeModel[layer + '_maskcode']
        # biases[layer] = quantizeModel[layer + '_bias']

    newstate_dict = model.state_dict()
    for k,v in model.state_dict().items():
        layer = k.split('.weight')[0]
        if layer in layers:
            newstate_dict[k] = torch.FloatTensor(codebook[layer][maskCode[layer]])

        # biaslayer = k.split('.bias')[0]
        # if biaslayer in layers:
        #     newstate_dict[k] = torch.FloatTensor(biases[biaslayer])

    model.load_state_dict(newstate_dict)

    codeDict = {}
    for layer in layers:
        a = maskCode[layer].flatten()
        b = range(len(a))
        codeDict[layer] = {}
        for i in range(len(a)):
            # codeDict保存每个码有哪些位置，而maskCode保存每个位置属于哪个码
            codeDict[layer].setdefault(a[i], []).append(b[i])

    return codebook,maskCode,codeDict

def retrainQuantizeModel(model,train_loader,optimizer,epoch,codebook,codeDict,maskCode,total_layers):
    model.train()
    for batch, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = loss_func(output, target)
        optimizer.zero_grad()
        loss.backward()
        grads = {}
        grads['conv1'] = model.conv1.weight.grad
        grads['conv2'] = model.conv2.weight.grad
        grads['fc1'] = model.fc1.weight.grad
        grads['fc2'] = model.fc2.weight.grad
        optimizer.step()
        updateCodebook(model, codebook, codeDict, maskCode, total_layers, grads)

        if batch % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                epoch, batch * len(data), len(train_loader.dataset),
                       100. * batch / len(train_loader), loss.item()))

def test(model, test_loader,codebook,maskCode):
    model.eval()
    test_loss = 0
    correct = 0
    global  best_val_loss
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
        best_val_loss = test_loss
        saveQuantizeModel(codebook, maskCode, 'QuantizeModel', total_layers)
        with open(os.path.join('./model', 'Quantize_Model.pt'), 'wb') as f:
            torch.save(model, f)

if __name__ == '__main__':
    with open(os.path.join('./model','Prune_Model.pt'),'rb') as f:
        model = torch.load(f,map_location='cpu')
    total_layers = ['conv1','conv2','fc1','fc2']
    num_c = 2 ** 8
    trainLoader = Data.DataLoader(dataset=torchvision.datasets.MNIST(
        root='./',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=False
    ), batch_size=64, shuffle=True)
    testLoader = Data.DataLoader(dataset=torchvision.datasets.MNIST(
        root='./',
        train=False,
        transform=torchvision.transforms.ToTensor()
    ), batch_size=100, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    codebook = {}
    maskCode = {}

    for i in range(10):
        codebook = kmeasModel(model, total_layers, num_c)
        codeDict, maskCode = quantizeModelwithDict(model, total_layers, codebook)
        retrainQuantizeModel(model,trainLoader,optimizer,i+1,codebook,codeDict,maskCode,total_layers)
        test(model,testLoader,codebook,maskCode)

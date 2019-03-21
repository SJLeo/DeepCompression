#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import pickle
import os
from LeNet import LeNet,DIYLeNet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--conv1_num",type= int)
parser.add_argument("--conv2_num",type=int)
args = parser.parse_args()

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

if __name__ == '__main__':
    total_layers = ['conv1', 'conv2', 'fc1', 'fc2']
    model = DIYLeNet(args.conv1_num,args.conv2_num)
    recover_all(model,total_layers,'QuantizeModel')

    with open(os.path.join('./model','Decode_Model.pt'),'wb') as f:
        torch.save(model,f)
    print('Decode done!')
import torch
import torchvision
import torch.nn as nn
import time
import torch.utils.data as Data
import os
from LeNet import LeNet,DIYLeNet

loss_func = nn.CrossEntropyLoss()

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    current_time = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += loss_func(output,target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss:{:4f}, Accuracy: {}/{} ({:.0f}% , Time:{:.3f}s)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset),time.time()-current_time
    ))

if __name__ == '__main__':
    with open(os.path.join('./model','Baseline_Model.pt'),'rb') as f:
        baselinemodel = torch.load(f,map_location='cpu')
    with open(os.path.join('./model','SSL_Model.pt'),'rb')as f:
        SSLmodel = torch.load(f,map_location='cpu')
    with open(os.path.join('./model','Prune_Model.pt'), 'rb') as f:
        prunemodel = torch.load(f, map_location='cpu')
    with open(os.path.join('./model','Quantize_Model.pt'), 'rb') as f:
        quantizemodel = torch.load(f, map_location='cpu')
    with open(os.path.join('./model','Decode_Model.pt'), 'rb') as f:
        decodemodel = torch.load(f, map_location='cpu')

    testLoader = Data.DataLoader(
        torchvision.datasets.MNIST(
            './',train=False,transform=torchvision.transforms.ToTensor()
        ),batch_size=100,shuffle=True
    )

    print('---------------------Baseline---------------------')
    test(baselinemodel, testLoader)
    print('-----------------------SSL------------------------')
    test(SSLmodel,testLoader)
    print('-----------------------Prune----------------------')
    test(prunemodel,testLoader)
    print('---------------------Quantize---------------------')
    test(quantizemodel,testLoader)
    print('----------------------Decode----------------------')
    test(decodemodel, testLoader)



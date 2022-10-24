import torch
from torch._C import dtype
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import random


EPOCH =  20
LR= 0.001
train_set = np.load('train_set.npy')
print(train_set.shape)
id_set = np.load('id_set.npy')

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=1),#(16,54,54)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2))#(16,27,27)
    
        self.layer2 = nn.Sequential( 
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1),#(32,26,26)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2))#(32,13,13)
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1),#(64,12,12)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2))#(64,5,5)

        
        self.fc = nn.Sequential(
            nn.Linear(64*5*5,20))
        
    def forward(self,x):
        x = self.layer1(x)  #(b,3,224,224),(3,8,4,4)——(b,8,111,111)  
        x = self.layer2(x)  #(b,8,111,111) ,(8,16,4,4) ——（b,16,55,55）——（b,16,28,28）
        x = self.layer3(x)  #(b,16,28,28）,(16,32,4,4)) ——(b,32,13,13)
        x = x.view(x.size(0),-1) #(b,64,3,3)——(b,64*3*3)
        x = self.fc(x)  #(b,64*3*3)*(64*3*3,256)——(b,256)——(b,256)*(256,64)——(b,64)——(b,64)*(64,20)——(b,20)
        return x 

cnn = CNN()
print(cnn)
optimizer = torch.optim.Adam(cnn.parameters(),lr=LR,)
loss_func = nn.CrossEntropyLoss()
test1 = np.zeros((8,56,56,3))
j = 0
for epoch in range(EPOCH):
    for i in range(8976):
        i = int(i)
        Id = np.zeros(4)
        Id = id_set[4*i:4*i+4]
        Id = torch.Tensor(Id).long()
        test1 = train_set[4*i:4*i+4]
        test1 = np.transpose(test1,(0,3,1,2))
        test1 = torch.tensor(test1,dtype=torch.float32)
        output = cnn(test1)
        loss = loss_func(output,Id)

        
        if j % 60 == 0:
            print('loss:',loss.data.numpy())
        j += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(cnn,'mycnn_20.pkl')
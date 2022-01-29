import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
from matplotlib import pyplot as plt 
import struct

class RMSELoss(nn.Module):
    
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, inputs, targets):        
        tmp = (inputs-targets)**2
        loss =  torch.mean(tmp)        
        return torch.sqrt(loss)


class Net_0(nn.Module):
    def __init__(self):
        super(Net_0, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        # self.fc1 = nn.Linear(768, 1) #16*12=16*(7-4)*(8-4)
        self.fc1 = nn.Linear(7*6*32, 1)
        self.drop1=nn.Dropout(p=0.1)
        self.drop2=nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
#         x = self.drop2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

class Net_L(nn.Module):
    def __init__(self):
        super(Net_L, self).__init__()
        self.fc1 = nn.Linear(7*6, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net_CNN(nn.Module):
    def __init__(self):
        super(Net_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(7*6*128, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8, 1)
        self.drop1=nn.Dropout(p=0.1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        # x = self.drop1(x)
        x = F.leaky_relu(self.conv2(x))
        # x = self.drop1(x)
        x = F.leaky_relu(self.conv3(x))
        # x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x))
        # x = self.drop1(x)
        x = F.leaky_relu(self.fc2(x))
        # x = self.drop1(x)
        x = F.leaky_relu(self.fc3(x))
        # x = self.drop1(x)
        x = self.fc4(x)
        return x


class Net_CNNFC(nn.Module):
    def __init__(self):
        super(Net_CNNFC, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.fc1 = nn.Linear(6*3*32+4, 1)


    def forward(self, x, y):
        # p=0.5
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = torch.cat((x,y),1)
        # x = F.relu(self.fc1(x))
        # x = self.fc3(x)
        print(x.shape)
        x = self.fc1(x)
        return x

class Net_tkzw(nn.Module):
    def __init__(self):
        super(Net_tkzw, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64*24, 32)
        self.fc0 = nn.Linear(32, 1)
        self.drop1=nn.Dropout(p=0.1)
        self.drop4=nn.Dropout(p=0.4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.leaky_relu(self.conv1(x))
        # x = F.leaky_relu(self.conv2(x))
        x = torch.flatten(x, 1)

        # x = F.leaky_relu(self.fc1(x))
        x = F.relu(self.fc1(x))
        # x = self.drop4(x)
        x = self.fc0(x)

        return x




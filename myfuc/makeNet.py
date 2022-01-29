import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
from torchvision import models

'''
Training
'''

class RMSELoss(nn.Module):
    
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, inputs, targets):        
        tmp = (inputs-targets)**2
        loss =  torch.mean(tmp)        
        return torch.sqrt(loss)


class Net_0(nn.Module):
    def __init__(self,inputSize):
        super(Net_0, self).__init__()
        self.inputSize=inputSize
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.fc1 = nn.Linear(self.inputSize*32, 1)


    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

class Net_L(nn.Module):
    def __init__(self,inputSize):
        super(Net_L, self).__init__()
        self.inputSize=inputSize
        self.fc1 = nn.Linear(inputSize, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        # x = F.leaky_relu(self.fc1(x))
        # x = F.leaky_relu(self.fc2(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, 1)
        x = self.fcs(x)
        return x


class Net_CNN_simp(nn.Module):
    def __init__(self,inputSize,p=1):
        super(Net_CNN_simp, self).__init__()
        self.inputSize=inputSize
        self.p=p/10
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.fc1 = nn.Linear(inputSize*64, 512)
        self.fc2 = nn.Linear(512, 1)
        self.drop=nn.Dropout(p=self.p)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x

class Net_CNN(nn.Module):
    def __init__(self,inputSize,p=1):
        super(Net_CNN, self).__init__()
        self.inputSize=inputSize
        self.p=p/10
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 512, 3, padding=1)
        self.fc1 = nn.Linear(inputSize*512, 64)
        self.fc2 = nn.Linear(64, 1)
        self.drop=nn.Dropout(p=self.p)


    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x

class Net_CNN_front(nn.Module):
    def __init__(self,inputSize,p=1):
        super(Net_CNN_front, self).__init__()
        self.inputSize=inputSize
        self.p=p/10
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 512, 3, padding=1)
        self.fc1 = nn.Linear(inputSize*512, 64)
        self.fc2 = nn.Linear(64, 1)
        self.drop=nn.Dropout(p=self.p)


    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )
def convLrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.LeakyReLU(inplace=True),
    )


class Net_UN(nn.Module):
    def __init__(self,inputSize):
        super().__init__()
        self.inputSize=inputSize
        self.out_channels=1

        self.conv1 = self.contract_block(1, 32, 3, 1)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        self.upconv1 = self.expand_block(32*2, self.out_channels, 3, 1)

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            # torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            # torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            # torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            # torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            # torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) 
                            )
        return expand


    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)
        print("upconv3,conv2 shape: ",upconv3.shape,conv2.shape)

        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))
        print("upconv1 shape: ",upconv1.shape)
        return upconv1

    


class Net_RUN(nn.Module):
    def __init__(self,inputSize):
        # super(Net_UN, self).__init__()
        super().__init__()
        self.inputSize=inputSize

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 1)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 1)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 1)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 1)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)


    def forward(self, x):
        x_original = self.conv_original_size0(x)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer4 = self.layer4_1x1(layer4)

        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)
        return x
    


class Net_CNNFC(nn.Module):
    def __init__(self,inputSize):
        super(Net_CNNFC, self).__init__()
        self.inputSize=inputSize
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.fc1 = nn.Linear(inputSize*32+4, 1)


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

class Net_tkzw_git(nn.Module):
    def __init__(self,inputSize):
        super(Net_tkzw_git, self).__init__()
        self.inputSize=inputSize
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(inputSize*64, 32)
        self.fc0 = nn.Linear(32, 2)
        self.drop4=nn.Dropout(p=0.4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop4(x)
        x = self.fc0(x)
        return x


class Net_tkzw2(nn.Module):
    def __init__(self,inputSize,kernel=64):
        super(Net_tkzw2, self).__init__()
        self.inputSize=inputSize
        self.kernel=kernel
        self.drop1=nn.Dropout(p=0.1)
        self.drop4=nn.Dropout(p=0.45)
        self.conv1 = nn.Conv2d(1, self.kernel, 3, padding=1)
        self.fc1 = nn.Linear(self.inputSize*self.kernel, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        # x = self.drop4(x)
        x = self.fc1(x)

        return x

class Net_L2(nn.Module):
    def __init__(self,inputSize,kernel=64):
        super(Net_L2, self).__init__()
        self.inputSize=inputSize
        self.kernel=kernel
        self.fc1 = nn.Linear(self.inputSize, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 2)
        self.drop1=nn.Dropout(0.1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


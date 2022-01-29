import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np

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
    
class MAELoss(nn.Module):
    
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, inputs, targets):        
        tmp = torch.abs(inputs-targets)
        loss =  torch.mean(tmp)        
        return loss
        
class ATANLoss(nn.Module):
    
    def __init__(self):
        super(ATANLoss, self).__init__()

    def forward(self, inputs, targets):        
        loss =  torch.mean(torch.atan(torch.abs(inputs-targets)))    
        return loss


def qloss(output, target,q=10):
    output=output.to("cpu").detach().numpy()
    target=target.to("cpu").detach().numpy()
    return np.sum(np.abs((output+0.5)//q-(target+0.5)//q))
    # return torch.sum(torch.abs((output+0.5)//q-(target+0.5)//q))

class trainS(object):
    def trainS1(self,NET,epoch,train_loader,vali_loader,criterion,optimizer):
        train_loss,vali_loss,vali_qloss = [0,0,0]
        for batch_idx, (data,target) in enumerate(train_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(),target.cuda()
            optimizer.zero_grad()
            output = NET(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        for batch_idx, (data, target) in enumerate(vali_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = NET(data)
            loss = criterion(output, target)
            vali_loss += loss.item()
            vali_qloss += qloss(output, target)

        print("Epoch %5d ||  Training %10f  ||  Validation %10f ||  Vali_Qloss %10f"
              %(epoch+1,train_loss / len(train_loader),vali_loss / len(vali_loader), vali_qloss))

        return [vali_loss / len(vali_loader),train_loss / len(train_loader), vali_qloss]

    def trainS1mae(self,NET,epoch,train_loader,vali_loader,criterion,optimizer):
        train_loss,vali_loss,vali_qloss = [0,0,0]
        for batch_idx, (data,target) in enumerate(train_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(),target.cuda()
            optimizer.zero_grad()
            output = NET(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        for batch_idx, (data, target) in enumerate(vali_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = NET(data)
            loss = criterion(output, target)
            vali_loss += loss.item()
            vali_qloss += qloss(output, target)


        print("Epoch %5d ||  Training %10f  ||  Validation %10f"
              %(epoch+1,train_loss / len(train_loader),vali_loss / len(vali_loader)))

        return vali_loss / len(vali_loader)

    def trainS2(self,NET,epoch,train_loader,vali_loader,criterion,optimizer):
        train_loss,vali_loss = [0,0]
        for batch_idx, (data,data2,target) in enumerate(train_loader):
            if torch.cuda.is_available():
                data, data2, target = data.cuda(),data2.cuda(), target.cuda()
            optimizer.zero_grad()
            output = NET(data,data2)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        for batch_idx, (data, data2, target) in enumerate(vali_loader):
            if torch.cuda.is_available():
                data, data2, target = data.cuda(),data2.cuda(), target.cuda()
            optimizer.zero_grad()
            output = NET(data,data2)
            loss = criterion(output, target)
            vali_loss += loss.item()

        print("Epoch %5d ||  Training %10f  ||  Validation %10f"
              %(epoch+1,train_loss / len(train_loader),vali_loss / len(vali_loader)))


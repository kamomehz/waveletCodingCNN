from itertools import count
# from msilib.schema import RadioButton
from pyexpat import ExpatError
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
from matplotlib import pyplot as plt 

'''
MAKE DATASET
'''

def rmse(a,b):
    return np.sqrt(((a-b)**2).mean())
def mse(a,b):
    return ((a-b)**2).mean()
def mae(a,b):
    return np.abs(a-b).mean()
def atan(a,b):
    return (np.arctan(np.abs(a-b))).mean()
def quan(a:"numpyArray",q=10):
    return np.round(a/q)

class makeDataset(Dataset):
    
    def __init__(self):
        self.tx=torch.Tensor(0)
        self.ty=torch.Tensor(0)
    
    def tkzw(self,low,high,ori,res,width=6,height=3):
        xs=[]
        label=[]
        ny,nx=ori.shape #256,512
        for j in range(ny-height):
            for i in range(0,nx-width+1,2): #0,2,...,502,504  Total 253
                ind1=(j+height,i+width//2-1)
                ind2=(j+height,slice(i//2,i//2+(1+width)//2))
                ind3=(slice(j,j+height),slice(i,i+width))
                label.append(ori[ind1])
                xs.extend(np.ravel(ori[ind3]))
                xs.extend(np.repeat(np.ravel(low[ind2]),2)[:]/2)

        label=np.array(label).reshape((-1,1))
        xs=np.array(xs).reshape((-1,1,height+1,width))
#         xs=np.array(xs).reshape((-1,1,width,height+1))
        print("data increase",label.size)
        self.tx = torch.cat((self.tx,torch.Tensor(xs)),0)
        self.ty = torch.cat((self.ty,torch.Tensor(label)),0)
        
    def tkzw2(self,low,high,ori,res,width=6,height=3):
        xs=[]
        label=[]
        ny,nx=ori.shape #256,512
        for j in range(ny-height):
            for i in range(0,nx-width+1,2): #0,2,...,502,504  Total 253
                ind1=(j+height,slice(i+width//2-1,i+width//2+1))
                ind2=(j+height,slice(i//2,i//2+(1+width)//2))
                ind3=(slice(j,j+height),slice(i,i+width))
                label.extend(ori[ind1])
                xs.extend(np.ravel(res[ind3]))
                xs.extend(np.repeat(np.ravel(low[ind2]),2)[:]/np.sqrt(2))
                # xs[-4:-2]=label[-2:]
        label=np.array(label).reshape((-1,2))
        xs=np.array(xs).reshape((-1,1,height+1,width))
#         xs=np.array(xs).reshape((-1,1,width,height+1))
        
        # print("data increase",label.shape[0])
        self.tx = torch.cat((self.tx,torch.Tensor(xs)),0)
        self.ty = torch.cat((self.ty,torch.Tensor(label)),0)

    def my(self,low,high,ori,res,width=7,height=5):
        xs=[]
        label=[]
        ny,nx=ori.shape #256,512
        for j in range(ny-height): # 0~252  Total 253
            for i in range(0,nx-width+1,2): #0,2,...,502,504  Total 253
                ind1=(j+height,(i+width)//2-2)
                ind2=(j+height,slice(i//2,i//2+(1+width)//2))
                ind3=(slice(j,j+height),slice(i,i+width))
                label.append(high[ind1])
                xs.extend(np.ravel(res[ind3]))
                xs.extend(np.repeat(np.ravel(low[ind2]),2)[:-1])

        label=np.array(label).reshape((-1,1))
        xs=np.array(xs).reshape((-1,1,height+1,width))
        # xs=np.array(xs).reshape((-1,1,width,height+1))
        # print("- size append ",label.size)
        self.tx = torch.cat((self.tx,torch.Tensor(xs)),0)
        self.ty = torch.cat((self.ty,torch.Tensor(label)),0)
    
    def myN(self,low,high,ori,res,width=7,height=5):
        xs=[]
        label=[]
        RATIO=np.max(np.abs(low))
        ny,nx=ori.shape #256,512
        temp=np.empty((height+1,width))
        # low,high,ori,res=self.plus0001(low,high,ori,res)
        for j in range(ny-height): # 0~252  Total 253
            for i in range(0,nx-width+1,2): #0,2,...,502,504  Total 253
                ind1=(j+height,(i+width)//2-2)
                ind2=(j+height,slice(i//2,i//2+(1+width)//2))
                ind3=(slice(j,j+height),slice(i,i+width))
                
                
                label.append(high[ind1]/RATIO)

                temp[:height,:]=res[ind3]/RATIO*2**0.5
                temp[height:height+1,:]=np.repeat(np.ravel(low[ind2]),2)[:-1]/RATIO
                temp=temp
                xs.extend(np.ravel(temp))
            
        print("RATIO=np.max(np.abs(low)) =",RATIO)        
        label=np.array(label).reshape((-1,1))
        xs=np.array(xs).reshape((-1,1,height+1,width))
        self.tx = torch.cat((self.tx,torch.Tensor(xs)),0)
        self.ty = torch.cat((self.ty,torch.Tensor(label)),0)

    def myN2(self,low,high,ori,res,width=7,height=3,normType=2):
        xs=[]
        label=[]
        norm=[]
        ny,nx=ori.shape #256,512
        RATIO=np.max(low)

        # normType=1
        for j in range(ny-height): # 0~252  Total 253
            for i in range(0,nx-width+1,2): #0,2,...,502,504  Total 253
                temp=np.empty((height+1,width))
                ind1=(j+height,(i+width)//2-2)
                ind2=(j+height,slice(i//2,i//2+(1+width)//2))
                ind3=(slice(j,j+height),slice(i,i+width))
                
                temp[:height,:]=res[ind3]
                temp[height:height+1,:]=(np.repeat(np.ravel(low[ind2]),2)[:-1])/2**0.5
                if normType==0:
                    RATIO= np.sum(np.abs(temp))
                    norm.append(RATIO)
                elif normType==1:
                    temp = (temp-np.min(temp))/(np.max(temp)-np.min(temp))
                    high[ind1] = high[ind1]/(np.max(temp)-np.min(temp))
                    RATIO=1
                elif normType==2:
                    pass
                elif normType==3:
                    RATIO=1

                # print("target=%03f, Rato=%03f"%(high[ind1]/RATIO,RATIO))

                label.append(high[ind1]/RATIO)
                xs.extend(np.ravel(temp/RATIO))
                
        label=np.array(label).reshape((-1,1))
        xs=np.array(xs).reshape((-1,1,height+1,width))
        self.tx = torch.cat((self.tx,torch.Tensor(xs)),0)
        self.ty = torch.cat((self.ty,torch.Tensor(label)),0)
        return np.array(norm)

    def plus0001(self,*arg):
        out=[]
        for array in arg:
            out.append(array+0.001)
        return out
    def minus0001(self,*arg):
        for array in arg:
            return array-0.001
        
    def plot(self,low,high,ori,res,s=10):
        print("ori.shape =",ori.shape)
        plt.figure(figsize=(s*2,s))
        plt.subplot(231)
        plt.title("low")
        plt.imshow(low,"gray")
        plt.subplot(232)
        plt.title("high")
        plt.imshow(high,"gray")
        plt.subplot(233)
        plt.title("res")
        plt.imshow(res,"gray")
        plt.subplot(234)
        plt.title("ori")
        plt.imshow(ori,"gray")
        plt.subplot(235)
        plt.title("ori-res")
        plt.imshow(ori-res,"gray")

    def out(self):
        print("Data amount =",self.ty.size()[0])
        return TensorDataset(self.tx,self.ty)
    
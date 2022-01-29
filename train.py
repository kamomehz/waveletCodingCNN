from torchvision import transforms
from PIL import Image
import os,glob
import re
from myfuc import *
import time
time.sleep(0*105*60)

dataLoader=readWTData()
datasetMaker=makeDataset()
print(torch.cuda.is_available())


w, h=7,4

# Training set
dataLoader=readWTData()
datasetMaker=makeDataset()
in_list=glob.glob("ppwOut_New_5_Q/train/*/")
in_ori="ppwOut_New_5_NOQ"
for path0 in in_list[:]:
    for lv in range(5):
        imgName=re.findall("\w+/",path0)[-1][:-1]
        
        pathLow=path0+"ewpLH-LCoefR_%d.dig"%lv
        pathHigh=path0+"ewpLH-HCoef_%d.dig"%lv
        pathRes=path0+"ewpLH-ImageR_%d.dig"%lv
        pathOri=in_ori+path0[path0.find("/"):]+"ewpLH-ImageR_%d.dig"%lv
        low,high,res,ori=dataLoader.makeImg(pathLow,pathHigh,pathRes,pathOri) # target is high
        datasetMaker.myN2(low,high,ori,res,width=w, height=h-1)
    
        pathLow=path0+"ewpHL-LCoefR_%d.dig"%lv
        pathHigh=path0+"ewpHL-HCoef_%d.dig"%lv
        pathRes=path0+"ewpHL-ImageR_%d.dig"%lv
        pathOri=in_ori+path0[path0.find("/"):]+"ewpHL-ImageR_%d.dig"%lv
        low,high,res,ori=dataLoader.makeImg(pathLow,pathHigh,pathRes,pathOri) # target is high
        datasetMaker.myN2(low,high,ori,res,width=w, height=h-1)

trainData=datasetMaker.out()
datasetMaker.plot(low,high,ori,res,5)

# Vali set
dataLoader=readWTData()
datasetMaker=makeDataset()
in_list=glob.glob("ppwOut_New_5_Q/val/*/")
in_list=['ppwOut_New_5_Q/val/BOAT/', 'ppwOut_New_5_Q/val/LENNA/', 'ppwOut_New_5_Q/val/BARBARA/']
in_ori="ppwOut_New_5_NOQ"
imgind=2
# for path0 in in_list:
for path0 in in_list[imgind:imgind+1]:
    for lv in range(5):
        imgName=re.findall("\w+/",path0)[-1][:-1]

        pathLow=path0+"ewpLH-LCoefR_%d.dig"%lv
        pathHigh=path0+"ewpLH-HCoef_%d.dig"%lv
        pathRes=path0+"ewpLH-ImageR_%d.dig"%lv
        pathOri=in_ori+path0[path0.find("/"):]+"ewpLH-ImageR_%d.dig"%lv
        low,high,res,ori=dataLoader.makeImg(pathLow,pathHigh,pathRes,pathOri) # target is high
        print("%10s high rmse:"%imgName,rmse(high,np.average(high)))
        datasetMaker.myN2(low,high,ori,res,width=w, height=h-1)
        
        pathLow=path0+"ewpHL-LCoefR_%d.dig"%lv
        pathHigh=path0+"ewpHL-HCoef_%d.dig"%lv
        pathRes=path0+"ewpHL-ImageR_%d.dig"%lv
        pathOri=in_ori+path0[path0.find("/"):]+"ewpHL-ImageR_%d.dig"%lv
        low,high,res,ori=dataLoader.makeImg(pathLow,pathHigh,pathRes,pathOri) # target is high
        print("%10s high rmse:"%imgName,rmse(high,0))
        datasetMaker.myN2(low,high,ori,res,width=w, height=h-1)

        # pathLow=path0+"ewpHH-LCoefR_%d.dig"%lv
        # pathHigh=path0+"ewpHH-HCoef_%d.dig"%lv
        # pathRes=path0+"ewpHH-ImageR_%d.dig"%lv
        # pathOri=in_ori+path0[path0.find("/"):]+"ewpHH-ImageR_%d.dig"%lv
        # low,high,res,ori=dataLoader.makeImg(pathLow,pathHigh,pathRes,pathOri) # target is high
        # print("%10s rmse:"%imgName,rmse(ori,res))
        # datasetMaker.myN2(low,high,ori,res,width=w, height=h-1)
    
valData=datasetMaker.out()
datasetMaker.plot(low,high,ori,res,5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# criterion=MAELoss()
criterion=RMSELoss()
# criterion=ATANLoss()
# criterion=QLoss()
criterion.to(device)
print(len(trainData),len(valData))

print("input_size: %d"%(w*h))
model=Net_CNN(w*h,4.5)
# model=Net_CNN_simp(w*h,3)

n_epochs = 100
log_interval,random_seed = 10,1
learning_rate=0.0002
betas0=(0.5,0.999)
batch_size = 16
batch_size = 64

torch.manual_seed(random_seed)
train_loader= DataLoader(trainData,batch_size=batch_size,shuffle=1)
vali_loader = DataLoader(valData,batch_size=batch_size,shuffle=1)
model = nn.DataParallel(model).to(device)
optimizer = optim.Adam(model.parameters(),lr=learning_rate,betas=betas0)
print(model)

trainer=trainS()
best=float("inf")
modelPath="my_74_2H_dp45_norm2_lr2_batch64/"
os.makedirs(modelPath, exist_ok=True)
losses=[]
for i in range(n_epochs):
    valLoss,trainLoss,qLoss=trainer.trainS1(model,i,train_loader,vali_loader,criterion,optimizer)
    checkpoint = {
        'state_dict': model.module.state_dict(),
        'opt_state_dict': optimizer.state_dict(),
        'epoch': i
    }
    losses.append(valLoss)
    losses.append(trainLoss)
    losses.append(qLoss)

    torch.save(checkpoint, modelPath+"checkpoint%03d.pt"%i)
    if valLoss<best:
        torch.save(checkpoint, modelPath+"best%03d.pt"%i)
        best=valLoss

plt.close()

plt.figure()
plt.plot(losses[1::3],label="Train")
plt.plot(losses[::3],label="Validation",color="r")
plt.xlabel("epoch")
plt.legend()
plt.show()
plt.savefig(modelPath+"epochs.png")


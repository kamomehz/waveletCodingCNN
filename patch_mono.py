import os
import glob
from time import sleep
from PIL import Image

#mono_p2pgm to ppwOut

root="/home/yy/kamome/Research/"
root="/export/work/kamomes/"
root="/export/work/kamomes/Research/"
base=root+"monoPGM/"
target_base=root+"ppwOut_New_5_NOQ/"
# target_base=root+"ppwOut_New_5_Q/"
# target_base=root+"ppwOut_New_Line/"
target_base="/home/yy/kamome/Research/nette/p2pgm/"
choice=["val/","train/"]
for mode in choice[:]:
    img_list=glob.glob(base+mode+"*")
    for timer,imgPath in enumerate(img_list[:]):
        imgName=imgPath.split("/")[-1][:-4] # .pgm==-4    .JPEG==-5
        with open(imgPath, "r") as file:
            read_xy = file.readline()
        nx,ny=read_xy.split(" ")[1:3]

        target_dir=target_base+mode
        targetPath=target_dir+imgName
        os.makedirs(targetPath, exist_ok=True)

        os.popen("cp ./ppwCoder ./ppwCoder_%s"%imgName)
        sleep(1)
        calla="./ppwCoder_%s %s %s %s %s"%(imgName,imgPath,targetPath,nx,ny)
        # calla=root+"Research/P1/ppwCoder %s %s %s %s"%(imgPath,targetPath,nx,ny)
        os.popen(calla)
        sleep(2)

print("Down!")      
sleep(2)
os.popen("rm ./ppwCoder_*")

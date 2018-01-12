import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import torchvision
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torch.optim as optim
import torch.nn.functional as F
from os.path import exists, join, basename, dirname
from os import makedirs, remove
import shutil
from torch.optim import lr_scheduler
import re
from dataloadercelebA import *
from AttrPreModelRes18_256V0 import *


torch.cuda.set_device(0)
cwd = os.getcwd()
print(cwd)

model = AttrPre()
model.cuda()
checkpoint = torch.load('AttrPreResNet18Det256V0_MSEloss.pth.tar', map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state_dict'])

with open("/home/miaoqianwen/FaceAttr/celebATest") as lmfile:
    lineNum=sum(1 for _ in lmfile)

it=iter(range(1, lineNum))
AttractiveCounter = 0
EyeGlassesCounter = 0
MaleCounter = 0
MouthOpenCounter = 0
SmilingCounter = 0
YoungCounter = 0
diff=0
for m in it:
    line = lc.getline("/home/miaoqianwen/FaceAttr/celebATest", m)
    line = line.rstrip('\n')
    file = line.split(' ')
    ImgName = file[0]
    iAttractive = []
    iEyeGlasses = []
    iMale = []
    iMouthOpen = []
    iSmiling = []
    iYoung = []
    iAttractive.append(float(file[3]))
    iEyeGlasses.append(float(file[16]))
    iMale.append(float(file[21]))
    iMouthOpen.append(float(file[22]))
    iSmiling.append(float(file[32]))
    iYoung.append(float(file[40]))
    iAttractive = np.asarray(iAttractive)
    iEyeGlasses = np.asarray(iEyeGlasses)
    iMale = np.asarray(iMale)
    iMouthOpen = np.asarray(iMouthOpen)
    iSmiling = np.asarray(iSmiling)
    iYoung = np.asarray(iYoung)
    input = io.imread(ImgName)
    if input.ndim < 3:
        input = cv2.cvtColor(input, cv2.COLOR_GRAY2RGB)
    inp = cv2.resize(input, (256, 256))
    imgI = (torch.from_numpy(inp.transpose((2, 0, 1))).float().div(255.0).unsqueeze_(0)-0.5)/0.5
    imgI = imgI.cuda()
    imgI = Variable(imgI)
    model.eval()
    AttractivePre, EyeGlassesPre, MalePre, MouthOpenPre, SmilingPre, YoungPre = model(imgI)
    AttractiveP = AttractivePre.cpu().data.numpy()[0]
    EyeGlassesP = EyeGlassesPre.cpu().data.numpy()[0]
    MaleP = MalePre.cpu().data.numpy()[0]
    MouthOpenP = MouthOpenPre.cpu().data.numpy()[0]
    SmilingP = SmilingPre.cpu().data.numpy()[0]
    YoungP = YoungPre.cpu().data.numpy()[0]
    if AttractiveP <0:
        if iAttractive[0] == -1:
            AttractiveCounter = AttractiveCounter +1
    else:
        if iAttractive[0] == 1:
            AttractiveCounter = AttractiveCounter +1

    if EyeGlassesP <0:
        if iEyeGlasses[0] == -1:
            EyeGlassesCounter = EyeGlassesCounter +1
    else:
        if iEyeGlasses[0] == 1:
            EyeGlassesCounter = EyeGlassesCounter +1

    if MaleP <0:
        if iMale[0] == -1:
            MaleCounter = MaleCounter +1
    else:
        if iMale[0] == 1:
            MaleCounter = MaleCounter +1

    if MouthOpenP <0:
        if iMouthOpen[0] == -1:
            MouthOpenCounter = MouthOpenCounter +1
    else:
        if iMouthOpen[0] == 1:
            MouthOpenCounter = MouthOpenCounter +1

    if SmilingP <0:
        if iSmiling[0] == -1:
            SmilingCounter = SmilingCounter +1
    else:
        if iSmiling[0] == 1:
            SmilingCounter = SmilingCounter +1

    if YoungP <0:
        if iYoung[0] == -1:
            YoungCounter = YoungCounter +1
    else:
        if iYoung[0] == 1:
            YoungCounter = YoungCounter +1

    print(ImgName)
    print("Attractive", ": ", iAttractive[0], " ", AttractiveP)
    print("EyeGlasses", ": ", iEyeGlasses[0], " ", EyeGlassesP)
    print("Male", ": ", iMale[0], " ", MaleP)
    print("MouthOpen", ": ", iMouthOpen[0], " ", MouthOpenP)
    print("Smiling", ": ", iSmiling[0], " ", SmilingP)
    print("Young", ": ", iYoung[0], " ", YoungP)
    #fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1)
    #ax.imshow(inp)
    #plt.show()

print(AttractiveCounter)
print(AttractiveCounter/lineNum)
print(EyeGlassesCounter)
print(EyeGlassesCounter/lineNum)
print(MaleCounter)
print(MaleCounter/lineNum)
print(MouthOpenCounter)
print(MouthOpenCounter/lineNum)
print(SmilingCounter)
print(SmilingCounter/lineNum)
print(YoungCounter)
print(YoungCounter/lineNum)

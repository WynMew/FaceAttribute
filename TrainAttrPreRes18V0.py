import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import torchvision
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from dataloadercelebA import *
from AttrPreModelRes18_256V0 import *
import torch.optim as optim
import torch.nn.functional as F
from os.path import exists, join, basename, dirname
from os import makedirs, remove
import shutil
from torch.optim import lr_scheduler
from sklearn.metrics import mutual_info_score
import math

datasetTrain = MyDataSet(filelist ='/home/miaoqianwen/FaceAttr/celebATrain',
            transform=transforms.Compose([
                ToTensorDict(),
                NormalizeImageDict(['image'])
            ]))
dataLoaderTrain = data.DataLoader(datasetTrain, batch_size=50, shuffle=True, num_workers=1)
datasetTest = MyDataSet(filelist ='/home/miaoqianwen/FaceAttr/celebAVal',
            transform=transforms.Compose([
                ToTensorDict(),
                NormalizeImageDict(['image'])
            ]))
dataLoaderTest = data.DataLoader(datasetTest, batch_size=50, shuffle=True, num_workers=1)

def add(x):
    with open('TrainAttrPreResNet18256V0Log',"a+") as outfile:
        outfile.write(x + "\n")

def save_checkpoint(state, is_best, file):
    model_dir = dirname(file)
    model_fn = basename(file)
    # make dir if needed (should be non-empty)
    if model_dir!='' and not exists(model_dir):
        makedirs(model_dir)
    torch.save(state, file)
    if is_best:
        shutil.copyfile(file, join(model_dir,'best_' + model_fn))

def train(epoch, model, loss_fn, optimizer, dataloader,log_interval=50):
    model.train()
    train_loss = 0
    for i_batch, sample_batched in enumerate(dataloader):
        optimizer.zero_grad()
        img, Attractive, EyeGlasses, Male, MouthOpen, Smiling, Young = Variable(sample_batched['image'].cuda()), \
                Variable(sample_batched['Attractive'], requires_grad=False),\
                Variable(sample_batched['EyeGlasses'], requires_grad=False), \
                Variable(sample_batched['Male'], requires_grad=False), \
                Variable(sample_batched['MouthOpen'], requires_grad=False), \
                Variable(sample_batched['Smiling'], requires_grad=False), \
                Variable(sample_batched['Young'], requires_grad=False)
        AttractiveF = Attractive.type((torch.FloatTensor))
        EyeGlassesF = EyeGlasses.type((torch.FloatTensor))
        MaleF = Male.type((torch.FloatTensor))
        MouthOpenF = MouthOpen.type((torch.FloatTensor))
        SmilingF = Smiling.type((torch.FloatTensor))
        YoungF = Young.type((torch.FloatTensor))
        AttractivePre, EyeGlassesPre, MalePre, MouthOpenPre, SmilingPre, YoungPre = model(img)
        lossAttractive = loss_fn(AttractivePre,AttractiveF.cuda())
        lossEyeGlasses = loss_fn(EyeGlassesPre, EyeGlassesF.cuda())
        lossMale = loss_fn(MalePre, MaleF.cuda())
        lossMouthOpen = loss_fn(MouthOpenPre, MouthOpenF.cuda())
        lossSmiling = loss_fn(SmilingPre, SmilingF.cuda())
        lossYoung = loss_fn(YoungPre, YoungF.cuda())
        loss = lossAttractive + lossEyeGlasses + lossMale + lossMouthOpen + lossSmiling + lossYoung
        loss.backward()
        optimizer.step()
        train_loss += loss.data.cpu().numpy()[0]
        if i_batch % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
                epoch, i_batch , len(dataloader),
                100. * i_batch / len(dataloader), loss.data[0]))
            line = "Train Epoch: " + str(epoch) + " " + str(100. * i_batch / len(dataloader)) + " " + str(loss.data[0])
            add(line)
    train_loss /= len(dataloader)
    print('Train set: Average loss: {:.4f}'.format(train_loss))
    line = "Train set: Average loss: " + str(train_loss)
    add(line)
    return train_loss

def test(model,loss_fn,dataloader):
    model.eval()
    test_loss = 0
    for i_batch, sample_batched in enumerate(dataloader):
        img, Attractive, EyeGlasses, Male, MouthOpen, Smiling, Young = Variable(sample_batched['image'].cuda()), \
                Variable(sample_batched['Attractive'], requires_grad=False), \
                Variable(sample_batched['EyeGlasses'], requires_grad=False), \
                Variable(sample_batched['Male'], requires_grad=False), \
                Variable(sample_batched['MouthOpen'], requires_grad=False), \
                Variable(sample_batched['Smiling'], requires_grad=False), \
                Variable(sample_batched['Young'],requires_grad=False)
        AttractiveF = Attractive.type((torch.FloatTensor))
        EyeGlassesF = EyeGlasses.type((torch.FloatTensor))
        MaleF = Male.type((torch.FloatTensor))
        MouthOpenF = MouthOpen.type((torch.FloatTensor))
        SmilingF = Smiling.type((torch.FloatTensor))
        YoungF = Young.type((torch.FloatTensor))
        AttractivePre, EyeGlassesPre, MalePre, MouthOpenPre, SmilingPre, YoungPre = model(img)
        lossAttractive = loss_fn(AttractivePre, AttractiveF.cuda())
        lossEyeGlasses = loss_fn(EyeGlassesPre, EyeGlassesF.cuda())
        lossMale = loss_fn(MalePre, MaleF.cuda())
        lossMouthOpen = loss_fn(MouthOpenPre, MouthOpenF.cuda())
        lossSmiling = loss_fn(SmilingPre, SmilingF.cuda())
        lossYoung = loss_fn(YoungPre, YoungF.cuda())
        loss = lossAttractive + lossEyeGlasses + lossMale + lossMouthOpen + lossSmiling + lossYoung
        test_loss += loss.data.cpu().numpy()[0]
    test_loss /= len(dataloader)
    print('Test set: Average loss: {:.4f}'.format(test_loss))
    line = "Test set: Average loss: " + str(test_loss)
    add(line)
    return test_loss


def adjust_lr(optimizer, epoch, maxepoch, init_lr, power = 0.9):
    lr = init_lr * (1-epoch/maxepoch)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


torch.cuda.set_device(2)
cwd = os.getcwd()
print(cwd)

model = AttrPre()
model.cuda()
init_lr = 1e-4
optimizer = optim.SGD(model.parameters(), lr= init_lr, momentum=0.5)

loss = nn.MSELoss()
best_test_loss = float("inf")
print('Starting training...')
resume = 0
start_epoch = 1
end_epoch = 20
if resume:
    checkpoint = torch.load('/home/miaoqianwen/HDD6/FaceAlignment/MyPoseNet/WebFaceAffineRegression3b1Mynet_MSEloss.pth.tar',
                            map_location=lambda storage, loc: storage)
    #checkpoint = torch.load('/home/miaoqianwen/HDD6/FaceAlignment/CNNGeometricPytorch/trained_models/best_pascal_checkpoint_adam_affine_grid_loss.pth.tar', map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    #model.parameters
    start_epoch = checkpoint['epoch']
    best_test_loss = checkpoint['best_test_loss']
    optimizer.load_state_dict(checkpoint['optimizer'])


for epoch in range(start_epoch, end_epoch + 1):
    train_loss = train(epoch, model, loss, optimizer, dataLoaderTrain,  log_interval=10)
    test_loss = test(model, loss, dataLoaderTest)
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    lr_now = adjust_lr(optimizer, epoch, end_epoch + 1, init_lr, power=10)
    print(lr_now)
    line = "lr_Now: " + str(lr_now)
    add(line)
    # remember best loss
    is_best = test_loss < best_test_loss
    best_test_loss = min(test_loss, best_test_loss)
    #checkpoint_name = os.path.join('AngleregRessionAlex_mse_loss.pth.tar')
    checkpoint_name = os.path.join('AttrPreResNet18Det256V0_MSEloss.pth.tar')
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_test_loss': best_test_loss,
        'optimizer': optimizer.state_dict(),
    }, is_best, checkpoint_name)

print('Done!')
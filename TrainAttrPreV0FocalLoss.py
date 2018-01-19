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
from sklearn.metrics import mutual_info_score
import math
from dataloadercelebACE import *
from AttrPreModelRes34_256V0CE import *
from FocalLoss import *

datasetTrain = MyDataSet(filelist ='celebATrain',
            transform=transforms.Compose([
                ToTensorDict(),
                NormalizeImageDict(['image'])
            ]))
dataLoaderTrain = data.DataLoader(datasetTrain, batch_size=50, shuffle=True, num_workers=1)
datasetTest = MyDataSet(filelist ='celebAVal',
            transform=transforms.Compose([
                ToTensorDict(),
                NormalizeImageDict(['image'])
            ]))
dataLoaderTest = data.DataLoader(datasetTest, batch_size=50, shuffle=True, num_workers=1)

def add(x):
    with open('TrainAttrPreResNet34256V0FocalLossLog',"a+") as outfile:
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
        AttractiveF = Attractive.type((torch.LongTensor))
        EyeGlassesF = EyeGlasses.type((torch.LongTensor))
        MaleF = Male.type((torch.LongTensor))
        MouthOpenF = MouthOpen.type((torch.LongTensor))
        SmilingF = Smiling.type((torch.LongTensor))
        YoungF = Young.type((torch.LongTensor))
        AttractivePre, EyeGlassesPre, MalePre, MouthOpenPre, SmilingPre, YoungPre = model(img)
        lossAttractive = loss_fn(AttractivePre, torch.squeeze(AttractiveF).cuda())
        lossEyeGlasses = loss_fn(EyeGlassesPre, torch.squeeze(EyeGlassesF).cuda())
        lossMale = loss_fn(MalePre, torch.squeeze(MaleF).cuda())
        lossMouthOpen = loss_fn(MouthOpenPre, torch.squeeze(MouthOpenF).cuda())
        lossSmiling = loss_fn(SmilingPre, torch.squeeze(SmilingF).cuda())
        lossYoung = loss_fn(YoungPre, torch.squeeze(YoungF).cuda())
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
        AttractiveF = Attractive.type((torch.LongTensor))
        EyeGlassesF = EyeGlasses.type((torch.LongTensor))
        MaleF = Male.type((torch.LongTensor))
        MouthOpenF = MouthOpen.type((torch.LongTensor))
        SmilingF = Smiling.type((torch.LongTensor))
        YoungF = Young.type((torch.LongTensor))
        AttractivePre, EyeGlassesPre, MalePre, MouthOpenPre, SmilingPre, YoungPre = model(img)
        lossAttractive = loss_fn(AttractivePre, torch.squeeze(AttractiveF).cuda())
        lossEyeGlasses = loss_fn(EyeGlassesPre, torch.squeeze(EyeGlassesF).cuda())
        lossMale = loss_fn(MalePre, torch.squeeze(MaleF).cuda())
        lossMouthOpen = loss_fn(MouthOpenPre, torch.squeeze(MouthOpenF).cuda())
        lossSmiling = loss_fn(SmilingPre, torch.squeeze(SmilingF).cuda())
        lossYoung = loss_fn(YoungPre, torch.squeeze(YoungF).cuda())
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

class OESM_CrossEntropy(nn.Module):
    def __init__(self, down_k=0.9, top_k=0.7):
        super(OESM_CrossEntropy, self).__init__()
        self.loss = nn.NLLLoss()
        self.down_k = down_k
        self.top_k = top_k
        self.softmax = nn.LogSoftmax()
        return
    def forward(self, input, target):
        softmax_result = self.softmax(input)
        loss = Variable(torch.Tensor(1).zero_())
        for idx, row in enumerate(softmax_result):
            gt = target[idx]
            pred = torch.unsqueeze(row, 0)
            cost = self.loss(pred, gt)
            loss = torch.cat((loss, cost.cpu()), 0)
        loss = loss[1:]
        loss_m = -loss
        if self.top_k == 1:
            valid_loss = loss
        index = torch.topk(loss_m, int(self.down_k * loss.size()[0]))
        loss = loss[index[1]]
        index = torch.topk(loss, int(self.top_k * loss.size()[0]))
        valid_loss = loss[index[1]]
        return torch.mean(valid_loss)


torch.cuda.set_device(3)
cwd = os.getcwd()
print(cwd)

model = AttrPre()
model.cuda()
init_lr = 1e-4

optimizer = optim.SGD(model.parameters(), lr= init_lr, momentum=0.5)
#optimizer = optim.SGD(model.classifier.parameters(), lr= init_lr, momentum=0.5)

#loss = nn.MSELoss()
#loss = nn.CrossEntropyLoss()
loss = FocalLoss(gamma=2)
best_test_loss = float("inf")
print('Starting training...')
start_epoch = 1
end_epoch = 30


for epoch in range(start_epoch, end_epoch + 1):
    train_loss = train(epoch, model, loss, optimizer, dataLoaderTrain,  log_interval=7)
    test_loss = test(model, loss, dataLoaderTest)
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    lr_now = adjust_lr(optimizer, epoch, end_epoch + 1, init_lr, power=4)
    #lr_now = adjust_lr(optimizer, epoch, end_epoch + 1, init_lr, power=6)
    print(lr_now)
    line = "lr_Now: " + str(lr_now)
    add(line)
    # remember best loss
    is_best = test_loss < best_test_loss
    best_test_loss = min(test_loss, best_test_loss)
    checkpoint_name = os.path.join('AttrPreResNet34Det256V0_FocalLoss.pth.tar')
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_test_loss': best_test_loss,
        'optimizer': optimizer.state_dict(),
    }, is_best, checkpoint_name)

print('Done!')

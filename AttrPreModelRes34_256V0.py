from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F


class FeatureExtraction(torch.nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()
        self.resnet = models.resnet34(pretrained=True)
        #self.resnet = models.resnet34(pretrained=False)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        # freeze parameters
        #for param in self.vgg.parameters():
        #    param.requires_grad = False
        # move to GPU
        self.resnet.cuda()

    def forward(self, image_batch):
        return self.resnet(image_batch)

class Classifier(nn.Module):
    def __init__(self, output_dim=1):
        super(Classifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, output_dim),
        )
        self.fc1.cuda()
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, output_dim),
        )
        self.fc2.cuda()
        self.fc3 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, output_dim),
        )
        self.fc3.cuda()
        self.fc4 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, output_dim),
        )
        self.fc4.cuda()
        self.fc5 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, output_dim),
        )
        self.fc5.cuda()
        self.fc6 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, output_dim),
        )
        self.fc6.cuda()

    def forward(self, x):
        x = x.view(x.size(0), -1) # flatten
        #print(x)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        x4 = self.fc4(x)
        x5 = self.fc5(x)
        x6 = self.fc6(x)
        return x1, x2, x3, x4, x5, x6

class AttrPre(nn.Module):
    def __init__(self):
        super(AttrPre, self).__init__()
        self.FeatureExtraction = FeatureExtraction()
        output_dim = 1
        self.classifier = Classifier(output_dim)

    def forward(self, img):
        # do feature extraction
        feature = self.FeatureExtraction(img)
        Attractive, EyeGlasses, Male, MouthOpen, Smiling, Young = self.classifier(feature)
        return Attractive, EyeGlasses, Male, MouthOpen, Smiling, Young
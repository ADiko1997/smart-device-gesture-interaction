import torch
import torchvision 
from torch.utils.data import DataLoader
import numpy as np 
import cv2 as cv 
import torch.nn.functional as F 
import torch.nn as nn

#This is the NN model
class Net(nn.Module):
    """
    Simple CNN architecture for grayscale images of hand gestures
    """
    def __init__(self):

        super(Net, self).__init__()
        self.cnn1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=[3,3],stride=1, padding=1)
        self.cnn2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=[3,3],stride=1, padding=1)
        self.cnn3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[3,3], padding=1)
        self.cnn4 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[3,3], padding=1)
        self.cnn5 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3,3], padding=1)
        self.cnn6 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3,3], padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=[3,3])
        self.Bnorm = torch.nn.BatchNorm2d(num_features=256)
        self.drop = torch.nn.Dropout2d(p=0.3)
        self.lin1 = torch.nn.Linear(256*7*7, out_features=512)
        self.lin2 = torch.nn.Linear(512, 50)
        self.lin3 = torch.nn.Linear(50, 6)

    def forward(self, x):

        x = F.relu(self.cnn1(x))
        x = self.pool(F.relu(self.cnn2(x)))
        x = F.relu(self.cnn3(x))
        x = self.pool(F.relu(self.cnn4(x)))
        x = F.relu(self.cnn5(x))
        x = self.pool(F.relu(self.cnn6(x)))
        x = self.Bnorm(x)
        x = self.drop(x)
        x = x.view(x.shape[0], 256*7*7)
        x = F.relu(self.lin1(x))
        x = self.drop(x)
        x = F.relu(self.lin2(x))  
        x = self.drop(x)
        x = self.lin3(x)
    
        return x
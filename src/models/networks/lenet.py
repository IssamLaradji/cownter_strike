
import torch
from torch import nn
import tqdm
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import os
import numpy as np
import time
from sklearn.metrics import confusion_matrix
import skimage
from haven import haven_utils as hu
from torchvision import transforms


class LeNeT(nn.Module):
    def __init__(self):
        super().__init__()
        nb_filters = 32
        nb_conv = 4
        self.nb_pool = 2

        self.conv1 = nn.Conv2d(1, nb_filters, (nb_conv,nb_conv),  padding=0)
        self.conv2 = nn.Conv2d(nb_filters, nb_filters, (nb_conv, nb_conv),  padding=0)
        # self.conv3 = nn.Conv2d(nb_filters, nb_filters*2, (nb_conv, nb_conv), 1)
        # self.conv4 = nn.Conv2d(nb_filters*2, nb_filters*2, (nb_conv, nb_conv), 1)

        self.dropout1 = nn.Dropout2d(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(3872, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x_input):
        n,c,h,w = x_input.shape
        x = self.conv1(x_input)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, self.nb_pool, self.nb_pool)
        x =  self.dropout1(x)

        x = x.view(n, -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x
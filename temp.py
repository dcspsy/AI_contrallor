import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F


class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.bn0 = nn.BatchNorm2d(4)
        self.con1 = nn.Conv2d(in_channels=4,out_channels=16, kernel_size=5,stride=1,padding=2,)
        # input shape (4, 9, 9) output shape (16, 9, 9)
        self.pool1 = nn.MaxPool2d(kernel_size=3)  # output shape (16, 3, 3)

        self.con2 = nn.Conv2d(16, 32, 3)  # input shape (16, 3, 3)output shape (32, 1, 1)
        self.out = nn.Linear(32, 1)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.bn0(x)
        x = F.relu(self.con1(x))
        x = self.pool1(x)
        x = F.relu(self.con2(x))
        x = x.view(x.size(0), -1)  # (batch_size, 32)
        output = self.out(x)
        return output

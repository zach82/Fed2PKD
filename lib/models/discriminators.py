import torch
import torchvision
import torch.nn as nn
import numpy as np

MNIST_PROTOTYPE_SIZE = 50

class D_MNIST(nn.Module):
    def __init__(self, args):
        super(D_MNIST, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(MNIST_PROTOTYPE_SIZE + args.num_classes, 512),
            torch.nn.GELU(),
            nn.Linear(512, 256),
            torch.nn.GELU(),
            nn.Linear(256, 64),
            torch.nn.GELU(),
            nn.Linear(64, 32),
            torch.nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    def forward(self, _input):
        prob = self.model(_input)
        return prob
    
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F
import torchvision.models as models


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)

class CNNFemnist(nn.Module):
    def __init__(self, args):
        super(CNNFemnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, args.out_channels, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(16820/20*args.out_channels), 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x1 = F.relu(self.fc1(x))
        x = F.dropout(x1, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), x1

class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, args.out_channels, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(320/20*args.out_channels), 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x1 = F.relu(self.fc1(x))
        x = F.dropout(x1, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), x1

class CNNFemnist_1(nn.Module):
    def __init__(self, args):
        super(CNNFemnist_1, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, args.out_channels, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(16820/20*args.out_channels), 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x1 = F.relu(self.fc1(x))
        x = F.dropout(x1, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), x1
    
class CNNFemnist_2(nn.Module):
    def __init__(self, args):
        super(CNNFemnist_2, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, args.out_channels, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(4704/4*args.out_channels), 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x1 = F.relu(self.fc1(x))
        x = F.dropout(x1, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), x1

class CNNFemnist_3(nn.Module):
    def __init__(self, args):
        super(CNNFemnist_3, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, args.out_channels, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(9408/4*args.out_channels), 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x1 = F.relu(self.fc1(x))
        x = F.dropout(x1, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), x1

CNNFemnistClasses = [CNNFemnist_1, CNNFemnist_2, CNNFemnist_3]

class CNNMnist_1(nn.Module):
    def __init__(self, args):
        super(CNNMnist_1, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, args.out_channels, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(320/20*args.out_channels), 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x1 = F.relu(self.fc1(x))
        x = F.dropout(x1, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), x1
    
class CNNMnist_2(nn.Module):
    def __init__(self, args):
        super(CNNMnist_2, self).__init__()
        # 定义第一个卷积层，输入通道数为args.num_channels，输出通道数为10，卷积核大小为5x5
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        # 定义第二个卷积层，输入通道数为10，输出通道数为20，卷积核大小为5x5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 定义第三个卷积层，输入通道数为20，输出通道数为args.out_channels，卷积核大小为3x3
        self.conv3 = nn.Conv2d(20, args.out_channels, kernel_size=3)
        # 定义dropout层，用于防止过拟合
        self.conv3_drop = nn.Dropout2d()
        # 定义第一个全连接层，输入节点数为1620/20*args.out_channels，输出节点数为50
        self.fc1 = nn.Linear(int(args.out_channels), 50)
        # 定义第二个全连接层，输入节点数为50，输出节点数为args.num_classes
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        # 第一个卷积层，使用ReLU激活函数和2x2的最大池化层
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # 第二个卷积层，使用ReLU激活函数和2x2的最大池化层
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # 第三个卷积层，使用ReLU激活函数、3x3的卷积核、2x2的最大池化层和dropout层
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        # 将张量展平为向量，用于全连接层
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x1 = F.relu(self.fc1(x))
        x = F.dropout(x1, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), x1

    
class CNNMnist_3(nn.Module):
    def __init__(self, args):
        super(CNNMnist_3, self).__init__()
        # 定义第一个卷积层，输入通道数为args.num_channels，输出通道数为10，卷积核大小为5x5
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        # 定义第二个卷积层，输入通道数为10，输出通道数为20，卷积核大小为5x5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 定义第三个卷积层，输入通道数为20，输出通道数为args.out_channels，卷积核大小为3x3
        self.conv3 = nn.Conv2d(20, args.out_channels, kernel_size=3)
        # 定义dropout层，用于防止过拟合
        self.conv3_drop = nn.Dropout2d()
        # 定义第一个全连接层，输入节点数为1620/20*args.out_channels，输出节点数为50
        self.fc1 = nn.Linear(int(args.out_channels), 50)
        # 定义第二个全连接层，输入节点数为50，输出节点数为25
        self.fc2 = nn.Linear(50, 25)
        # 定义第三个全连接层，输入节点数为25，输出节点数为args.num_classes
        self.fc3 = nn.Linear(25, args.num_classes)

    def forward(self, x):
        # 第一个卷积层，使用ReLU激活函数和2x2的最大池化层
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # 第二个卷积层，使用ReLU激活函数和2x2的最大池化层
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # 第三个卷积层，使用ReLU激活函数、3x3的卷积核、2x2的最大池化层和dropout层
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        # 将张量展平为向量，用于全连接层
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x1 = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x1))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), x1

CNNMnistClasses = [CNNMnist_1, CNNMnist_2, CNNMnist_3]

class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc0 = nn.Linear(16 * 5 * 5, 120)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, args.num_classes)

    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = x.view(-1, 16 * 5 * 5)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return F.log_softmax(x, dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x1 = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x1))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), x1


class Lenet(nn.Module):
    def __init__(self, args):
        super(Lenet, self).__init__()
        self.n_cls = 10
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, self.n_cls)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x1 = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x1))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1), x1
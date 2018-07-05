# -*- coding:utf-8 -*-
# Created Time: Thu 05 Jul 2018 10:00:41 PM CST
# Author: Taihong Xiao <xiaotaihong@126.com>

import torch
import torchvision
import os
import argparse
use_cuda = True

def generator(dataloader):
    while True:
        for data in dataloader:
            yield data

resnet50 = torchvision.models.resnet50(pretrained=False)
resnet50.load_state_dict(torch.load('./models/resnet50-19c8e357.pth'))

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_set = torchvision.datasets.MNIST('./datasets/mnist/', train=True, transform=None, download=False)
test_set = torchvision.datasets.MNIST('./datasets/mnist/', train=False, transform=None, download=False)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True, **kwargs)
# data_generator = generator(dataloader)

for batch_idx, (data, target) in enumerate(train_loader):
    # data = next(data_generator)
    if batch_idx < 1:
        print(data)



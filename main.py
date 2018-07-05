# -*- coding:utf-8 -*-
# Created Time: Thu 05 Jul 2018 10:00:41 PM CST
# Author: Taihong Xiao <xiaotaihong@126.com>

import torch
import torchvision
import os
import argparse


def generator(dataloader):
    while True:
        for data in dataloader:
            yield data

resnet50 = torchvision.models.resnet50(pretrained=False)
resnet50.load_state_dict(torch.load('./models/resnet50-19c8e357.pth'))


dataset = torchvision.datasets.MNIST('./datasets/mnist/', train=True, transform=None, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
data_generator = generator(dataloader)

for i in range(1):
    data = next(data_generator)
    print(data)



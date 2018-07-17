# -*- coding:utf-8 -*-
# Created Time: Thu 05 Jul 2018 10:00:41 PM CST
# Author: Taihong Xiao <xiaotaihong@126.com>

import torch
import torchvision
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import os
import argparse

train_dir = 'train_log'
cuda = True
batch_size = 30
w1, h1 = 224, 224
w2, h2 = 28, 28
lmd = 5e-5
lr = 0.005
decay = 0.99
max_epoch = 30
restore = False

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

def imagenet_label2_mnist_label(imagenet_label):
    return imagenet_label[:,:10]

def tensor2var(tensor, requires_grad=False, cuda=False, volatile=False):
    if cuda:
        with torch.cuda.device(0):
            tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad, volatile=volatile)
    return var

def generator(dataloader):
    while True:
        for data in dataloader:
            yield data

transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
])


resnet50 = torchvision.models.resnet50(pretrained=False)
resnet50.load_state_dict(torch.load('./models/resnet50-19c8e357.pth'))
resnet50.eval()

kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}
train_set = torchvision.datasets.MNIST('./datasets/mnist/', train=True, transform=transform, download=False)
test_set = torchvision.datasets.MNIST('./datasets/mnist/', train=False, transform=transform, download=False)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, **kwargs)
# train_generator = generator(train_loader)
# test_generator = generator(test_loader)

# test image read

# from PIL import Image
# import time

# t1 = time.time()
# img1 = np.asanyarray(Image.open('dog.jpg').resize((224,224)))
# img1 = np.transpose(img1, (2,0,1)) / 255.
# t2 = time.time()

# import cv2
# img = cv2.cvtColor(cv2.imread('cat1.jpg'), cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, (224,224))
# img = np.transpose(img, (2,0,1)) / 255.


# mean = np.array([0.485, 0.456, 0.406])
# mean = mean[..., np.newaxis, np.newaxis]
# std = np.array([0.229, 0.224, 0.225])
# std = std[..., np.newaxis, np.newaxis]

# img = (img - mean) / std
# img = img[np.newaxis, ...].astype(np.float32)
# x = tensor2var(torch.from_numpy(img))
# out = resnet50(x)
# print(np.argmax(out.data.numpy()))
# from IPython import embed; embed();


# mean and std for input
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
mean = mean[..., np.newaxis, np.newaxis]
std = np.array([0.229, 0.224, 0.225],dtype=np.float32)
std = std[..., np.newaxis, np.newaxis]
mean = tensor2var(torch.from_numpy(mean), cuda=cuda)
std = tensor2var(torch.from_numpy(std), cuda=cuda)

# create mask M
M = np.ones((3, h1, w1), dtype=np.float32)
c_w, c_h = int(np.ceil(w1/2.)), int(np.ceil(h1/2.))
M[:,c_h-h2//2:c_h+h2, c_w-w2//2:c_w+w2//2] = 0
M = tensor2var(torch.from_numpy(M), cuda=cuda)

# Learnable parameter W
if restore:
    W = torch.load(os.path.join(train_dir, 'W_{:03d}.pt'.format(restore))).data
else:
    W = torch.randn(M.shape)

W = tensor2var(W, requires_grad=True, cuda=cuda)

# optimizer
BCE = torch.nn.BCELoss()
optimizer = torch.optim.Adam([W], lr=lr, betas=(0.5, 0.999))
# optimizer = torch.optim.SGD([W], lr=lr)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=decay)

if cuda:
    with torch.cuda.device(0):
        BCE.cuda()
        resnet50.cuda()


# start training
for i in range(max_epoch):
    for j, (image, label) in enumerate(train_loader):
        lr_scheduler.step()
        image = np.tile(image, (1,3,1,1))
        label = torch.zeros(batch_size, 10).scatter_(1, label.view(-1,1), 1)
        label = tensor2var(label, cuda=cuda)

        X = np.zeros((batch_size, 3, h1, w1), dtype=np.float32)
        X[:,:,(h1-h2)//2:(h1+h2)//2, (w1-w2)//2:(w1+w2)//2] = image
        X = tensor2var(torch.from_numpy(X), cuda=cuda)

        P = torch.sigmoid(W * M)
        X_adv = X + P # range [0, 1]
        X_adv = (X_adv - mean) / std

        Y_adv = resnet50(X_adv)
        Y_adv = F.softmax(Y_adv, 1)
        out = imagenet_label2_mnist_label(Y_adv)
        loss = BCE(out, label) #+ lmd * torch.norm(W) ** 2
        # loss =  lmd * torch.norm(W) ** 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(torch.norm(W).data.cpu().numpy())
        print('epoch %03d/%03d, batch %06d, loss %.6f' % (i + 1, max_epoch, j + 1, loss.data.cpu().numpy()))

    # test
    acc = 0.0
    for j, (image, label) in enumerate(test_loader):
        image = np.tile(image, (1,3,1,1))
        # label = torch.zeros(batch_size, 10).scatter_(1, label.view(-1,1), 1)
        # label = tensor2var(label)

        X = torch.zeros(batch_size, 3, h1, w1)
        X[:,:,(h1-h2)//2:(h1+h2)//2, (w1-w2)//2:(w1+w2)//2] = torch.from_numpy(image)
        X = tensor2var(X)
        if cuda:
            with torch.cuda.device(0):
                X = X.cuda()
        P = torch.sigmoid(W * M)
        X_adv = X + P # range [0, 1]

        Y_adv = resnet50(X_adv)
        Y_adv = F.softmax(Y_adv, 1)
        out = imagenet_label2_mnist_label(Y_adv)
        pred = out.data.cpu().numpy().argmax(1)
        acc += sum(label.numpy() == pred) / float(len(label) * len(test_loader))
    print('epoch %03d/%03d, test accuracy %.6f' % (i, max_epoch, acc))



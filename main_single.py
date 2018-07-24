# -*- coding:utf-8 -*-
# Created Time: Thu 05 Jul 2018 10:00:41 PM CST
# Author: Taihong Xiao <xiaotaihong@126.com>
from config import cfg

import torch
import torchvision
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import os
import argparse
from tqdm import tqdm, trange


class Adversarial_Reprogramming(object):
    def __init__(self, args, cfg=cfg):
        self.mode = args.mode
        self.gpu = args.gpu
        self.restore = args.restore
        self.cfg = cfg
        self.init_net()
        self.init_dataset()
        self.init_mask()
        self.init_weight()
        self.set_mode_and_gpu()

    def init_net(self):
        if self.cfg.net == 'resnet50':
            self.net = torchvision.models.resnet50(pretrained=False)
            self.net.load_state_dict(os.path.join(self.cfg.models_dir, 'resnet50-19c8e357.pth'))

            # mean and std for input
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            mean = mean[..., np.newaxis, np.newaxis]
            std = np.array([0.229, 0.224, 0.225],dtype=np.float32)
            std = std[..., np.newaxis, np.newaxis]
            self.mean = self.tensor2var(torch.from_numpy(mean))
            self.std = self.tensor2var(torch.from_numpy(std))

        elif self.cfg.net == 'vgg16':
            self.net = torchvision.models.vgg16(pretrained=False)
            self.net.load_state_dict(torch.load(os.path.join(self.cfg.models_dir, 'vgg16-397923af.pth')))

            # mean and std for input
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            mean = mean[..., np.newaxis, np.newaxis]
            std = np.array([0.229, 0.224, 0.225],dtype=np.float32)
            std = std[..., np.newaxis, np.newaxis]
            self.mean = Parameter(torch.from_numpy(mean), requires_grad=False)
            self.std = Parameter(torch.from_numpy(std), requires_grad=False)

        else:
            raise NotImplementationError()

        self.net.eval()

    def init_dataset(self):
        if self.cfg.dataset == 'mnist':
            train_set = torchvision.datasets.MNIST(os.path.join(self.cfg.data_dir, 'mnist'), train=True, transform=transforms.ToTensor(), download=True)
            test_set = torchvision.datasets.MNIST(os.path.join(self.cfg.data_dir, 'mnist'), train=False, transform=transforms.ToTensor(), download=True)
            kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}
            self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.cfg.batch_size, shuffle=True, **kwargs)
            self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.cfg.batch_size, shuffle=True, **kwargs)
        else:
            raise NotImplementationError()

    def init_mask(self):
        M = np.ones((3, self.cfg.h1, self.cfg.w1), dtype=np.float32)
        c_w, c_h = int(np.ceil(self.cfg.w1/2.)), int(np.ceil(self.cfg.h1/2.))
        M[:,c_h-self.cfg.h2//2:c_h+self.cfg.h2, c_w-self.cfg.w2//2:c_w+self.cfg.w2//2] = 0
        self.M = self.tensor2var(torch.from_numpy(M))

    def init_weight(self):
        # Learnable parameter W
        if self.restore:
            W = torch.load(os.path.join(self.cfg.train_dir, 'W_{:03d}.pt'.format(self.restore))).data
            self.start_epoch = self.restore + 1
        else:
            W = torch.randn(self.M.shape)
            self.start_epoch = 1
        self.W = self.tensor2var(W, requires_grad=True)

    def set_mode_and_gpu(self):
        if self.mode == 'train':
            # optimizer
            self.BCE = torch.nn.BCELoss()
            self.optimizer = torch.optim.Adam([self.W], lr=self.cfg.lr, betas=(0.5, 0.999))
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=self.cfg.decay)
            if self.gpu:
                with torch.cuda.device(self.gpu[0]):
                    self.BCE.cuda()
                    self.net.cuda()

            # if len(self.gpu) > 1:
            #     self.W = torch.nn.DataParallel(self.W, device_ids=self.gpu)

        elif self.mode == 'test':
            if self.gpu:
                with torch.cuda.device(self.gpu[0]):
                    self.net.cuda()

            # if len(self.gpu) > 1:
            #     self.W = torch.nn.DataParallel(self.W, device_ids=self.gpu)

        else:
            raise NotImplementationError()

    def imagenet_label2_mnist_label(self, imagenet_label):
        return imagenet_label[:,:10]

    def tensor2var(self, tensor, requires_grad=False, volatile=False):
        if self.gpu:
            with torch.cuda.device(self.gpu[0]):
                tensor = tensor.cuda()
        return Variable(tensor, requires_grad=requires_grad, volatile=volatile)

    def forward(self, image):
        image = np.tile(image, (1,3,1,1))
        X = np.zeros((self.cfg.batch_size, 3, self.cfg.h1, self.cfg.w1), dtype=np.float32)
        X[:,:,(self.cfg.h1-self.cfg.h2)//2:(self.cfg.h1+self.cfg.h2)//2, (self.cfg.w1-self.cfg.w2)//2:(self.cfg.w1+self.cfg.w2)//2] = image
        X = self.tensor2var(torch.from_numpy(X))

        P = torch.sigmoid(self.W * self.M)
        X_adv = X + P # range [0, 1]
        X_adv = (X_adv - self.mean) / self.std

        Y_adv = self.net(X_adv)
        Y_adv = F.softmax(Y_adv, 1)
        return self.imagenet_label2_mnist_label(Y_adv)

    def compute_loss(self, out, label):
        label = torch.zeros(self.cfg.batch_size, 10).scatter_(1, label.view(-1,1), 1)
        label = self.tensor2var(label)
        return self.BCE(out, label) + self.cfg.lmd * torch.norm(self.W) ** 2

    def validate(self):
        acc = 0.0
        for i, (image, label) in enumerate(self.test_loader):
            out = self.forward(image)
            pred = out.data.cpu().numpy().argmax(1)
            acc += sum(label.numpy() == pred) / float(len(label) * len(self.test_loader))
        print('test accuracy: %.6f' % acc)

    def train(self):
        for i in range(self.start_epoch, self.cfg.max_epoch + 1):
            self.epoch = i
            self.lr_scheduler.step()
            for j, (image, label) in tqdm(enumerate(self.train_loader)):
                self.out = self.forward(image)
                self.loss = self.compute_loss(self.out, label)
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
            print('epoch: %03d/%03d, loss: %.6f' % (self.epoch, self.cfg.max_epoch, self.loss.data.cpu().numpy()))
            torch.save(self.W.cpu(), os.path.join(self.cfg.train_dir, 'W_%03d.pt' % self.epoch))
            self.validate()

    def test(self):
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default='train', type=str, choices=['train', 'test'])
    parser.add_argument('-r', '--restore', default=None, action='store', type=int, help='Specify checkpoint id to restore.')
    parser.add_argument('-g', '--gpu', default=[], nargs='+', type=str, help='Specify GPU ids.')
    # test params

    args = parser.parse_args()
    # print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)
    AR = Adversarial_Reprogramming(args)
    if args.mode == 'train':
        AR.train()
    elif args.mode == 'validate':
        AR.validate()
    elif args.mode == 'test':
        AR.test()
    else:
        raise NotImplementationError()

if __name__ == "__main__":
    main()

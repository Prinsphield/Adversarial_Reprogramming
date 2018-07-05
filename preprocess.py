# -*- coding:utf-8 -*-
# Created Time: Thu 05 Jul 2018 10:02:44 PM CST
# Author: Taihong Xiao <xiaotaihong@126.com>

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os,time
from PIL import Image

import codecs

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()

def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)

root = './datasets/mnist'
raw_folder = 'raw'
training_file = 'training.pt'
test_file = 'test.pt'

training_set = (
	read_image_file(os.path.join(root, raw_folder, 'train-images-idx3-ubyte.gz')),
	read_label_file(os.path.join(root, raw_folder, 'train-labels-idx1-ubyte.gz'))
)

test_set = (
	read_image_file(os.path.join(root, raw_folder, 't10k-images-idx3-ubyte.gz')),
	read_label_file(os.path.join(root, raw_folder, 't10k-labels-idx1-ubyte.gz'))
)

with open(os.path.join(root, processed_folder, training_file), 'wb') as f:
	torch.save(training_set, f)
with open(os.path.join(root, processed_folder, test_file), 'wb') as f:
	torch.save(test_set, f)

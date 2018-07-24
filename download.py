#!/usr/bin/env python
# Created Time: Thu 19 Jul 2018 09:13:02 PM CST
# Author: Taihong Xiao <xiaotaihong@126.com>

import subprocess
from config import cfg

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
}

for model_name in model_urls:
    print('Downloading {} model ...'.format(model_name))
    cmd = 'wget {} -P {}'.format(model_urls[model_name], cfg.models_dir)
    subprocess.call(cmd, shell=True)


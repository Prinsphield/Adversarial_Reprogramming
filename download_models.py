# -*- coding:utf-8 -*-
# Created Time: Thu 19 Jul 2018 09:13:02 PM CST
# Author: Taihong Xiao <xiaotaihong@126.com>
import os
from config import cfg

cmd = 'wget https://download.pytorch.org/models/resnet50-19c8e357.pth -P {}'.format(cfg.models_dir)
os.system(cmd)


import os
from easydict import EasyDict

cfg = EasyDict()

cfg.net = 'resnet50'
cfg.dataset = 'mnist'

cfg.train_dir = 'train_log'
cfg.models_dir = 'models'
cfg.data_dir = 'datasets'

cfg.batch_size = 240
cfg.w1 = 224
cfg.h1 = 224
cfg.w2 = 28
cfg.h2 = 28
cfg.lmd = 5e-7
cfg.lr = 0.05
cfg.decay = 0.96
cfg.max_epoch = 300

if not os.path.exists(cfg.train_dir):
    os.makedirs(cfg.train_dir)

if not os.path.exists(cfg.models_dir):
    os.makedirs(cfg.models_dir)

if not os.path.exists(cfg.data_dir):
    os.makedirs(cfg.data_dir)


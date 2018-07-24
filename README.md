# Adversarial Reprogramming

## Introduction

This repo is a pytorch implementation of the [paper](https://arxiv.org/abs/1806.11146).

- The `main_raw.py` is the rough version from scratch.
- The `main_single.py` is the old version that is only able to train on a single gpu card.
- The `main.py` is the final version that supports multi-gpu training.

## Requirements

- [Python 2.7 or 3.x](https://www.python.org/)
- [Pytorch 0.3](http://pytorch.org/)
- [tqdm](https://github.com/tqdm/tqdm)

## Training

0. Download pretrained imagenet models.

```
./download.py
```




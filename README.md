# Adversarial Reprogramming

## Introduction

This repo is a pytorch implementation of the [paper](https://arxiv.org/abs/1806.11146).
We aim to reprogram the pretrained imagenet models for MNIST classification.
Following the code, you can easily add more datasets and other pretrained imagenet models
for more experiments.

- The `main_raw.py` is the rough version from scratch.
- The `main_single.py` is the old version that is only able to train on a single gpu card.
- The `main.py` is the final version that supports multi-gpu training.

## Requirements

- [Python 2.7 or 3.x](https://www.python.org/)
- [Pytorch 0.3](http://pytorch.org/)
- [tqdm](https://github.com/tqdm/tqdm)

## Training

1. Download pretrained imagenet models.
    ```
    ./download.py
    ```
2. The MNIST dataset will be automatically downloaded after running the scripts.
   Then, the structure of this repo should be as follows.
    ```
    ├── datasets
    │   └── mnist
    │       ├── processed
    │       │   ├── test.pt
    │       │   └── training.pt
    │       └── raw
    │           ├── t10k-images-idx3-ubyte.gz
    │           ├── t10k-labels-idx1-ubyte.gz
    │           ├── train-images-idx3-ubyte.gz
    │           └── train-labels-idx1-ubyte.gz
    ├── models
    │   ├── resnet50-19c8e357.pth
    │   └── vgg16-397923af.pth
    ├── train_log
    │   └── W_001.pt
    ├── README.md
    ├── download.py
    ├── main.py
    ├── ....

    ```

3. So directly run the following scripts to train the model.
    ```
    python main.py -m train -g 0 1
    ```
    We provide three three modes, `train`, `validate` and `test`. The argment `-g` indicates the gpu ids.
    Add `-r 3` if you want to continue training your model from the third epoch.


## Validate

Simply running the following command.
```
python main.py -m validate -g 0
```

## Results




import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

import torchvision.utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import numpy as np
import os


if __name__ == "__main__":
    img_dir = "./data/jamo"
    img_data = dsets.ImageFolder(img_dir, transforms.Compose([
        transforms.Grayscale(),
        # https://pytorch.org/docs/stable/torchvision/transforms.html
        #           Data Augmentation
        #           transforms.RandomRotation(15)
        #           transforms.CenterCrop(28),
        #           transforms.Lambda(lambda x: x.rotate(15)),

        #           Data Nomalization
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
        # Normalize a tensor image with mean and standard deviation.
        # Given mean: (M1,...,Mn) and std: (S1,..,Sn) for n channels,
        # this transform will normalize each channel of the input torch.
        # *Tensor i.e. input[channel] = (input[channel] - mean[channel]) / std[channel]
    ]))

    # https://pytorch.org/docs/stable/torchvision/transforms.html

    print(img_data.classes)
    print(img_data.class_to_idx)
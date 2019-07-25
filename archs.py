# -*- coding: utf-8 -*-
import numpy as np

from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
import pretrainedmodels


class ResNet34(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.model = pretrainedmodels.resnet34(pretrained='imagenet')
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.model.last_linear = nn.Linear(
            in_features=512, out_features=num_classes, bias=True)

        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def forward(self, x):
        x_out = self.model(x)
        return x_out

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pretrainedmodels
from efficientnet_pytorch import EfficientNet


def get_model(model_name='resnet18', num_outputs=None, pretrained=True,
              freeze_bn=False, dropout_p=0, **kwargs):

    if 'efficientnet' in model_name:
        model = EfficientNet.from_pretrained(model_name, num_classes=num_outputs)

    elif 'densenet' in model_name:
        model = models.__dict__[model_name](num_classes=1000,
                                            pretrained=pretrained)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_outputs)

    else:
        pretrained = 'imagenet' if pretrained else None
        model = pretrainedmodels.__dict__[model_name](num_classes=1000,
                                                      pretrained=pretrained)

        if 'dpn' in model_name:
            in_channels = model.last_linear.in_channels
            model.last_linear = nn.Conv2d(in_channels, num_outputs,
                                          kernel_size=1, bias=True)
        else:
            if 'resnet' in model_name:
                model.avgpool = nn.AdaptiveAvgPool2d(1)
            else:
                model.avg_pool = nn.AdaptiveAvgPool2d(1)
            in_features = model.last_linear.in_features
            if dropout_p == 0:
                model.last_linear = nn.Linear(in_features, num_outputs)
            else:
                model.last_linear = nn.Sequential(
                    nn.Dropout(p=dropout_p),
                    nn.Linear(in_features, num_outputs),
                )

    if freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    return model

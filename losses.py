import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, focusing_param=2, balance_param=0.25):
        super(FocalLoss, self).__init__()
        self.focusing_param = focusing_param
        self.balance_param = balance_param

    def forward(self, output, target):
        cross_entropy = F.cross_entropy(output, target)
        cross_entropy_log = torch.log(cross_entropy)
        logpt = - F.cross_entropy(output, target)
        pt = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.focusing_param) * logpt
        balanced_focal_loss = self.balance_param * focal_loss

        return balanced_focal_loss

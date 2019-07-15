import torch
from sklearn import metrics
import numpy as np


def quadratic_weighted_kappa(y_pred, y_true):
    if torch.is_tensor(y_pred):
        y_pred = y_pred.data.cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.data.cpu().numpy()
    y_pred = np.argmax(y_pred, axis=1)
    return metrics.cohen_kappa_score(y_pred, y_true, weights='quadratic')

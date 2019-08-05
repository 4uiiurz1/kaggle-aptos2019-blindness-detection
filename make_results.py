import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, train_test_split
from skimage.io import imread

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from lib.dataset import Dataset
from lib.models.model_factory import get_model
from lib.utils import *
from lib.metrics import *
from lib.losses import *
from lib.preprocess import preprocess


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')

    args = parser.parse_args()

    return args


def main():
    test_args = parse_args()

    args = joblib.load('models/%s/args.pkl' %test_args.name)

    folds = []
    losses = []
    scores = []
    for fold in range(args.n_splits):
        log_path = 'models/%s/log_%d.csv' %(args.name, fold+1)
        if not os.path.exists(log_path):
            continue
        log = pd.read_csv('models/%s/log_%d.csv' %(args.name, fold+1))
        loss, score = log.loc[log['val_loss'].values.argmin(), ['val_loss', 'val_score']].values
        print(loss, score)
        folds.append(str(fold+1))
        losses.append(loss)
        scores.append(score)
    results = pd.DataFrame({
        'fold': folds + ['mean'],
        'loss': losses + [np.mean(losses)],
        'score': scores + [np.mean(scores)],
    })
    print(results)
    results.to_csv('models/%s/results.csv' % args.name, index=False)


if __name__ == '__main__':
    main()

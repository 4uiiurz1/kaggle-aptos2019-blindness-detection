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
    parser.add_argument('--tta', default=False, type=str2bool)

    args = parser.parse_args()

    return args


def apply_tta(input):
    inputs = []
    inputs.append(input)
    inputs.append(torch.flip(input, dims=[2]))
    inputs.append(torch.flip(input, dims=[3]))
    inputs.append(torch.rot90(input, k=1, dims=[2, 3]))
    inputs.append(torch.rot90(input, k=2, dims=[2, 3]))
    inputs.append(torch.rot90(input, k=3, dims=[2, 3]))
    inputs.append(torch.rot90(torch.flip(input, dims=[2]), k=1, dims=[2, 3]))
    inputs.append(torch.rot90(torch.flip(input, dims=[2]), k=3, dims=[2, 3]))
    return inputs


def main():
    test_args = parse_args()

    args = joblib.load('models/%s/args.pkl' %test_args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')

    if args.pred_type == 'classification':
        num_outputs = 5
    elif args.pred_type == 'regression':
        num_outputs = 1
    else:
        raise NotImplementedError

    cudnn.benchmark = True

    test_transform = transforms.Compose([
        transforms.Resize((args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # data loading code
    test_dir = preprocess(
        'test',
        args.img_size,
        scale=args.scale_radius,
        norm=args.normalize,
        pad=args.padding,
        remove=args.remove)
    test_df = pd.read_csv('inputs/test.csv')
    test_img_paths = test_dir + '/' + test_df['id_code'].values + '.png'
    test_labels = np.zeros(len(test_img_paths))

    test_set = Dataset(
        test_img_paths,
        test_labels,
        transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4)

    preds = []
    for fold in range(args.n_splits):
        print('Fold [%d/%d]' %(fold+1, args.n_splits))

        # create model
        model_path = 'models/%s/model_%d.pth' % (args.name, fold+1)
        if not os.path.exists(model_path):
            print('%s is not exists.' %model_path)
            continue
        model = get_model(model_name=args.arch,
                          num_outputs=num_outputs,
                          freeze_bn=args.freeze_bn,
                          dropout_p=args.dropout_p)
        model = model.cuda()
        model.load_state_dict(torch.load(model_path))

        model.eval()

        preds_fold = []
        with torch.no_grad():
            for i, (input, _) in tqdm(enumerate(test_loader), total=len(test_loader)):
                if test_args.tta:
                    outputs = []
                    for input in apply_tta(input):
                        input = input.cuda()
                        output = model(input)
                        outputs.append(output.data.cpu().numpy()[:, 0])
                    preds_fold.extend(np.mean(outputs, axis=0))
                else:
                    input = input.cuda()
                    output = model(input)

                    preds_fold.extend(output.data.cpu().numpy()[:, 0])
        preds_fold = np.array(preds_fold)
        preds.append(preds_fold)

        if not args.cv:
            break

    preds = np.mean(preds, axis=0)

    if test_args.tta:
        args.name += '_tta'

    test_df['diagnosis'] = preds
    test_df.to_csv('probs/%s.csv' %args.name, index=False)

    thrs = [0.5, 1.5, 2.5, 3.5]
    preds[preds < thrs[0]] = 0
    preds[(preds >= thrs[0]) & (preds < thrs[1])] = 1
    preds[(preds >= thrs[1]) & (preds < thrs[2])] = 2
    preds[(preds >= thrs[2]) & (preds < thrs[3])] = 3
    preds[preds >= thrs[3]] = 4
    preds = preds.astype('int')

    test_df['diagnosis'] = preds
    test_df.to_csv('submissions/%s.csv' %args.name, index=False)


if __name__ == '__main__':
    main()

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

from dataset import Dataset
import archs
from utils import *
from metrics import *
from losses import *
from preprocess import preprocess

arch_names = archs.__dict__.keys()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='ResNet34',
                        choices=arch_names,
                        help='model architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: ResNet34)')
    parser.add_argument('--loss', default='CrossEntropyLoss',
                        choices=['CrossEntropyLoss', 'FocalLoss', 'MSELoss'])
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--img_size', default=224, type=int,
                        help='input image size (default: 224)')
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--pred_type', default='classification',
                        choices=['classification', 'regression'])
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau'])
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--min_lr', default=1e-4, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.5, type=float)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    parser.add_argument('--cv', default=True, type=str2bool)
    parser.add_argument('--n_splits', default=5, type=int)

    # preprocessing
    parser.add_argument('--scale_radius', default=True, type=str2bool)
    parser.add_argument('--normalize', default=True, type=str2bool)
    parser.add_argument('--padding', default=False, type=str2bool)
    parser.add_argument('--remove', default=False, type=str2bool)

    # data augmentation
    parser.add_argument('--rotate', default=180, type=int)
    parser.add_argument('--rescale', default=0, type=float)
    parser.add_argument('--shear', default=36, type=int)
    parser.add_argument('--translate', default=0, type=float)
    parser.add_argument('--flip', default=True, type=str2bool)
    parser.add_argument('--random_erase', default=False, type=str2bool)
    parser.add_argument('--random_erase_prob', default=0.5, type=float)
    parser.add_argument('--random_erase_sl', default=0.02, type=float)
    parser.add_argument('--random_erase_sh', default=0.4, type=float)
    parser.add_argument('--random_erase_r', default=0.3, type=float)

    args = parser.parse_args()

    return args


def train(args, train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    scores = AverageMeter()

    model.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        if args.pred_type == 'classification':
            loss = criterion(output, target)
        elif args.pred_type == 'regression':
            loss = criterion(output.view(-1), target.float())

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.pred_type == 'regression':
            thrs = [0.5, 1.5, 2.5, 3.5]
            output[output < thrs[0]] = 0
            output[(output >= thrs[0]) & (output < thrs[1])] = 1
            output[(output >= thrs[1]) & (output < thrs[2])] = 2
            output[(output >= thrs[2]) & (output < thrs[3])] = 3
            output[output >= thrs[3]] = 4
        score = quadratic_weighted_kappa(output, target)

        losses.update(loss.item(), input.size(0))
        scores.update(score, input.size(0))

    return losses.avg, scores.avg


def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            if args.pred_type == 'regression':
                target = target.float()

            output = model(input)
            if args.pred_type == 'classification':
                loss = criterion(output, target)
            elif args.pred_type == 'regression':
                loss = criterion(output.view(-1), target.float())

            if args.pred_type == 'regression':
                thrs = [0.5, 1.5, 2.5, 3.5]
                output[output < thrs[0]] = 0
                output[(output >= thrs[0]) & (output < thrs[1])] = 1
                output[(output >= thrs[1]) & (output < thrs[2])] = 2
                output[(output >= thrs[2]) & (output < thrs[3])] = 3
                output[output >= thrs[3]] = 4
            score = quadratic_weighted_kappa(output, target)

            losses.update(loss.item(), input.size(0))
            scores.update(score, input.size(0))

    return losses.avg, scores.avg


def main():
    args = parse_args()

    if args.name is None:
        args.name = '%s_%s' %(args.arch, datetime.now().strftime('%m%d%H'))

    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    if args.loss == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss().cuda()
    elif args.loss == 'FocalLoss':
        criterion = FocalLoss().cuda()
    elif args.loss == 'MSELoss':
        criterion = nn.MSELoss().cuda()
    else:
        raise NotImplementedError

    if args.pred_type == 'classification':
        num_classes = 5
    elif args.pred_type == 'regression':
        num_classes = 1
    else:
        raise NotImplementedError

    cudnn.benchmark = True

    # data loading code
    dir_name = 'processed/train_images_%d' %args.img_size
    if args.scale_radius:
        dir_name += '_scaled'
    if args.normalize:
        dir_name += '_normed'
    if args.padding:
        dir_name += '_pad'
    if args.remove:
        dir_name += '_rm'
    if not os.path.exists(dir_name):
        preprocess(
            args.img_size,
            scale=args.scale_radius,
            norm=args.normalize,
            pad=args.padding,
            remove=args.remove)

    df = pd.read_csv('inputs/train.csv')
    img_paths = dir_name + '/' + df['id_code'].values + '.png'

    labels = df['diagnosis'].values

    train_img_paths, val_img_paths, train_labels, val_labels = \
        train_test_split(img_paths, labels, stratify=labels, test_size=1/args.n_splits, random_state=41)

    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomAffine(
            degrees=(-args.rotate, args.rotate),
            translate=(args.translate, args.translate),
            scale=(1.0 - args.rescale, 1.0 + args.rescale),
            shear=(-args.shear, args.shear),
        ),
        transforms.RandomHorizontalFlip(p=0.5 if args.flip else 0),
        transforms.RandomVerticalFlip(p=0.5 if args.flip else 0),
        RandomErase(
            prob=args.random_erase_prob if args.random_erase else 0,
            sl=args.random_erase_sl,
            sh=args.random_erase_sh,
            r=args.random_erase_r),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    best_losses = []
    best_scores = []

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=41)
    for fold, (train_idx, val_idx) in enumerate(skf.split(img_paths, labels)):
        print('Fold [%d/%d]' %(fold+1, args.n_splits))

        train_img_paths, val_img_paths = img_paths[train_idx], img_paths[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]

        train_set = Dataset(
            train_img_paths,
            train_labels,
            transform=train_transform)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4)

        val_set = Dataset(
            val_img_paths,
            val_labels,
            transform=val_transform)
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4)

        # create model
        model = archs.__dict__[args.arch](num_classes=num_classes)
        model = model.cuda()

        # print(model)

        if args.optimizer == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

        if args.scheduler == 'CosineAnnealingLR':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
        elif args.scheduler == 'ReduceLROnPlateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience,
                verbose=1, min_lr=args.min_lr)

        log = pd.DataFrame(index=[], columns=[
            'epoch', 'loss', 'score', 'val_loss', 'val_score'
        ])

        best_loss = float('inf')
        best_score = 0
        for epoch in range(args.epochs):
            print('Epoch [%d/%d]' %(epoch+1, args.epochs))

            # train for one epoch
            train_loss, train_score = train(args, train_loader, model, criterion, optimizer, epoch)
            # evaluate on validation set
            val_loss, val_score = validate(args, val_loader, model, criterion)

            if args.scheduler == 'CosineAnnealingLR':
                scheduler.step()
            elif args.scheduler == 'ReduceLROnPlateau':
                scheduler.step(val_loss)

            print('loss %.4f - score %.4f - val_loss %.4f - val_score %.4f'
                %(train_loss, train_score, val_loss, val_score))

            tmp = pd.Series([
                epoch,
                train_loss,
                train_score,
                val_loss,
                val_score
            ], index=['epoch', 'loss', 'score', 'val_loss', 'val_score'])

            log = log.append(tmp, ignore_index=True)
            log.to_csv('models/%s/log_%d.csv' %(args.name, fold+1), index=False)

            if val_loss < best_loss:
                torch.save(model.state_dict(), 'models/%s/model_%d.pth' %(args.name, fold+1))
                best_loss = val_loss
                best_score = val_score
                print("=> saved best model")

        print('val_loss:  %f' %best_loss)
        print('val_score: %f' %best_score)

        best_losses.append(best_loss)
        best_scores.append(best_score)

    results = pd.DataFrame({
        'fold': np.arange(1, args.n_splits+1),
        'best_loss': best_losses,
        'best_score': best_scores,
    })
    print(results)
    results.to_csv('models/%s/results.csv' %args.name, index=False)


if __name__ == '__main__':
    main()

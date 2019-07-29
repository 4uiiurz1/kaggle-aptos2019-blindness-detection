import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import cv2
import numpy as np
import random
import scipy.ndimage as ndi
from tqdm import tqdm
import os
from PIL import Image
from skimage.io import imread


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img_path, label = self.img_paths[index], self.labels[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = imread(img_path)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


    def __len__(self):
        return len(self.img_paths)

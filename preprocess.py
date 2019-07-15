import os
from glob import glob
import numpy as np
import cv2
from skimage import measure
import pandas as pd
from tqdm import tqdm


def scale_radius(src, img_size, padding=False):
    x = src[src.shape[0] // 2, ...].sum(axis=1)
    r = (x > x.mean() / 10).sum() // 2
    yx = src.sum(axis=2)
    region_props = measure.regionprops((yx > yx.mean() / 10).astype('uint8'))
    yc, xc = np.round(region_props[0].centroid).astype('int')
    x1 = max(xc - r, 0)
    x2 = min(xc + r, src.shape[1] - 1)
    y1 = max(yc - r, 0)
    y2 = min(yc + r, src.shape[0] - 1)
    dst = src[y1:y2, x1:x2]
    dst = cv2.resize(dst, dsize=None, fx=img_size/(2*r), fy=img_size/(2*r))
    if padding:
        pad_x = (img_size - dst.shape[1]) // 2
        pad_y = (img_size - dst.shape[0]) // 2
        dst = np.pad(dst, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), 'constant')
    return dst


def normalize(src, img_size):
    dst = cv2.addWeighted(src, 4, cv2.GaussianBlur(src, (0, 0), img_size / 30), -4, 128)
    return dst


def remove_boundaries(src, img_size):
    mask = np.zeros(src.shape)
    cv2.circle(
        mask,
        center=(src.shape[1] // 2, src.shape[0] // 2),
        radius=int(img_size / 2 * 0.9),
        color=(1, 1, 1),
        thickness=-1)
    dst = src * mask + 128 * (1 - mask)
    return dst


def preprocess(img_size, scale=False, norm=False, pad=False, remove=False):
    df = pd.read_csv('inputs/train.csv')
    img_paths = 'inputs/train_images/' + df['id_code'].values + '.png'

    dir_name = 'processed/train_images_%d' %img_size
    if scale:
        dir_name += '_scaled'
    if norm:
        dir_name += '_normed'
    if pad:
        dir_name += '_pad'
    if remove:
        dir_name += '_rm'
    os.makedirs(dir_name, exist_ok=True)
    for i in tqdm(range(len(img_paths))):
        img_path = img_paths[i]
        img = cv2.imread(img_path)
        if scale:
            img = scale_radius(img, img_size=img_size, padding=pad)
        img = cv2.resize(img, (img_size, img_size))
        if norm:
            img = normalize(img, img_size=img_size)
        if remove:
            img = remove_boundaries(img, img_size=img_size)
        cv2.imwrite(os.path.join(dir_name, os.path.basename(img_path)), img)


if __name__ == '__main__':
    import os
    from tqdm import tqdm
    import pandas as pd

    df = pd.read_csv('inputs/train.csv')
    img_paths = 'inputs/train_images/' + df['id_code'].values + '.png'

    os.makedirs('processed/train_images_224', exist_ok=True)
    for i in tqdm(range(len(img_paths))):
        img_path = img_paths[i]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        cv2.imwrite('processed/train_images_224/' + os.path.basename(img_path), img)


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#
#     img = cv2.imread('inputs/train_images/000c1434d8d7.png')
#     img = scale_radius(img, img_size=224, padding=True)
#     img = normalize(img, img_size=224)
#
#     plt.imshow(img)
#     plt.show()

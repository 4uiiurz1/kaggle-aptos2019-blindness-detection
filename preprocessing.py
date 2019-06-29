from glob import glob
import numpy as np
import cv2
from skimage import measure


def scale_radius(src, size, padding=False):
    x = src[src.shape[0] // 2, ...].sum(axis=1)
    r = (x > x.mean() / 10).sum() // 2
    yx = img.sum(axis=2)
    region_props = measure.regionprops((yx > yx.mean() / 10).astype('uint8'))
    yc, xc = np.round(region_props[0].centroid).astype('int')
    x1 = max(xc - r, 0)
    x2 = min(xc + r, src.shape[1] - 1)
    y1 = max(yc - r, 0)
    y2 = min(yc + r, src.shape[0] - 1)
    dst = src[y1:y2, x1:x2]
    dst = cv2.resize(dst, dsize=None, fx=size/(2*r), fy=size/(2*r))
    if padding:
        pad_x = (size - dst.shape[1]) // 2
        pad_y = (size - dst.shape[0]) // 2
        dst = np.pad(dst, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), 'constant')

    return dst


def normalize(src, size):
    dst = cv2.addWeighted(src, 4, cv2.GaussianBlur(img, (0, 0), size / 30), -4, 128)
    return dst


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = cv2.imread('inputs/train_images/000c1434d8d7.png')
    img = scale_radius(img, size=224, padding=True)
    img = normalize(img, size=224)

    plt.imshow(img)
    plt.show()

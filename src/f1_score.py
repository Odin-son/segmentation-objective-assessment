import cv2
import numpy as np
import os

ROOT_DIR = os.getcwd()
GT_DIR = os.path.join(ROOT_DIR, "../data/GT")
UNET_DIR = os.path.join(ROOT_DIR, "../data/UNET")
NESTEDUNET_DIR = os.path.join(ROOT_DIR, "../data/NESTEDUNET")
SPADEUNET_DIR = os.path.join(ROOT_DIR, "../data/SPADEUNET")


def findF1(value, gt):
    right = np.sum(value * gt == 1)
    precision = right / np.sum(value)
    recall = right / np.sum(gt)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def main():
    for i in range(1, 6):
        gt = cv2.imread(os.path.join(GT_DIR, '{}.png'.format(i)))[:, :, 0] / 255
        unet = cv2.imread(os.path.join(UNET_DIR, '{}.png'.format(i)))[:, :, 0] / 255
        nested = cv2.imread(os.path.join(NESTEDUNET_DIR, '{}.png'.format(i)))[:, :, 0] / 255
        spade = cv2.imread(os.path.join(SPADEUNET_DIR, '{}.png'.format(i)))[:, :, 0] / 255

        unet_f1 = findF1(unet.reshape(-1, ), gt.reshape(-1, ))
        nested_f1 = findF1(nested.reshape(-1, ), gt.reshape(-1, ))
        spade_f1 = findF1(spade.reshape(-1, ), gt.reshape(-1, ))

        print('[{}] unet : {}, nested : {}, spade : {}'.format(i, unet_f1, nested_f1, spade_f1))


if __name__ == '__main__':
    main()

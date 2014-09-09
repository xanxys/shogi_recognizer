#!/bin/python
from __future__ import print_function, division
import cv
import cv2
import os
import os.path


if __name__ == '__main__':
    dir_path = './dataset/no-mochigoma-initial'
    for p in os.listdir(dir_path):
        img_path = os.path.join(dir_path, p)
        img = cv2.imread(img_path)
        print('processing %s: shape=%s' % (img_path, img.shape))

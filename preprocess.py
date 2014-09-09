#!/bin/python
from __future__ import print_function, division
import numpy as np
import math
import cv
import cv2
import os
import os.path


def detect_board(photo_id, img, visualize):
    """
    * photo_id: str
    * img: BGR image
    Take color image and detect 9x9 black grid in shogi board
    It's assumed that shogi board occupies large portion of img.

    return: False failure
    """
    assert(len(img.shape) == 3)  # y, x, channel
    assert(img.shape[2] == 3)  # channel == 3
    # Resize image to keep height <= max_height.
    max_height = 1000
    if img.shape[0] > max_height:
        height, width = img.shape[:2]
        new_size = (int(width * max_height / height), max_height)
        img = cv2.resize(img, new_size)
    # Apply threshold to try to keep only grids
    # Be generous with noise though, because if grid is faint and
    # gone here, it will be impossible to detect grid in later steps.
    img_gray = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    img_bin = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 5)
    if visualize:
        cv2.imwrite('debug/letter-spacing: %s-binary.png' % photo_id, img_bin)
    # Detect lines. None or [[(rho,theta)]]
    num_lines_target = 500
    vote_thresh = 500
    change_rate = 0.8
    for i in range(5):
        lines = cv2.HoughLines(img_bin, 2, 0.01, int(vote_thresh))
        n_lines = 0 if lines is None else len(lines[0])
        print(i, n_lines)
        if n_lines < num_lines_target * 0.7:
            vote_thresh *= change_rate
        elif n_lines > num_lines_target * 1.3:
            vote_thresh /= change_rate
        else:
            break
        change_rate = (change_rate + 1) / 2
    assert(lines is not None)
    lines = lines[0]
    if visualize:
        img_gray_w_lines = cv2.cvtColor(img_gray, cv.CV_GRAY2BGR)
        for (rho, theta) in lines:
            line_d = np.array([-math.sin(theta), math.cos(theta)])
            line_n = np.array([math.cos(theta), math.sin(theta)])
            p0 = tuple((line_n * rho - line_d * 2000).astype(int))
            p1 = tuple((line_n * rho + line_d * 2000).astype(int))
            cv2.line(img_gray_w_lines, p0, p1, (0, 0, 255))
        cv2.imwrite('debug/letter-spacing: %s-raw-lines.png' % photo_id, img_gray_w_lines)

    # Find almost square 9x9 grid (lines are 10x10)
    return True



if __name__ == '__main__':
    dir_path = './dataset/no-mochigoma-initial'
    count = {
        "loaded": 0,
        "detected": 0
    }
    for (photo_id, p) in enumerate(os.listdir(dir_path)):
        img_path = os.path.join(dir_path, p)
        img = cv2.imread(img_path)
        print('processing %s: id=%s shape=%s' % (img_path, photo_id, img.shape))
        count["loaded"] += 1
        detected = detect_board(str(photo_id), img, True)
        if detected:
            count["detected"] += 1
        else:
            print('->failed')

    print(count)

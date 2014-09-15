#!/bin/python
from __future__ import print_function, division
import cv
import cv2
import argparse
import preprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Analyze shogi board state in a photo""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'photo', metavar='PHOTO', nargs=1, type=str,
        help='Photo image path')
    parser.add_argument(
        '--output-visualization', nargs='?', metavar='VISUALIZATION_PATH',
        type=str, default=None, const=True,
        help='Output path of pretty visualization image')

    args = parser.parse_args()
    img = cv2.imread(args.photo[0])
    # TODO: Refactoring required
    args.derive_emptiness = False
    args.derive_types_up = False
    args.derive_validness = False

    detected = preprocess.detect_board("", img, visualize=False, derive=args)
    print("Detected?: %s" % detected)

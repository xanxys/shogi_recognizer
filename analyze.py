#!/bin/python
from __future__ import print_function, division
import cv
import cv2
import argparse
import preprocess
import numpy as np


class RecognizedBoard(object):
    def __init__(self):
        pass

    def getPrecision(self):
        """
        Return probability [0, 1] of result being correct.
        """
        return 0

    def getCorners(self):
        """
        Return 4 corners of grid in image coordinates in CCW
        (as numpy array of shape (4, 2))
        (top-right, top-left, bottom-left, botttom-right)

        It's assumed the cell nearest to top-right corner is (1,1)
        and cell nearest to top-left is (9, 1) and so on.
        """
        return np.zeros([4, 2], float)

    def getState(self):
        """
        Return recognized board state as sparse dict of
        (cell location, piece state). Empty cells will not
        be included.

        key: (1,1)- (9,9)
        value: ("FU", "up"), ("NK", "down"), ...
        """
        return dict()

    def getCells(self):
        """
        Return image 4 corners for each cell in CCW order
        (like getCorners method, but for cells)

        Return value is always dict with 9^2 elements.
        """
        return dict()


class FakeDeriveFlags(object):
    """
    TODO:
    This is terrible hack. remove it immediately by refactoring.
    """
    pass


class BoardAnalyzer(object):
    """
    Initialize board analyzer by loading parameter files.
    """
    def __init__(self):
        pass

    def analyze(self, image):
        """
        Detect single shogi board in image and return RecognizedBoard.
        return None when recognition failed.

        When image contains multiple boards, return arbitrary one of them.
        (not recommended due to low precision)

        image: RGB image containing single shogi board
        return: RecognizedBoard or None
        """
        derive = FakeDeriveFlags()
        derive.derive_emptiness = False
        derive.derive_types_up = False
        derive.derive_validness = False
        det = preprocess.detect_board("", image, visualize=False, derive=derive)
        if det:
            return RecognizedBoard()
        else:
            return None


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

    analyzer = BoardAnalyzer()
    detected = analyzer.analyze(img)
    print("Detected: %s" % detected)

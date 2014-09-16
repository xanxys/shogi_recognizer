#!/bin/python
from __future__ import print_function, division
import cairo
import cv
import cv2
import argparse
import preprocess
import numpy as np
import math


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
        if det is not None:
            print(det["corners"])
            board = RecognizedBoard()
            board.corners = det["corners"]
            return board
        else:
            return None


def to_cairo_surface(img):
    """
    Convert OpenCV BGR image to cairo.ImageSurface
    """
    h, w, c = img.shape
    assert(c == 3)
    # BGR -> BGRX(little endian)
    arr = np.zeros([h, w, 4], np.uint8)
    arr[:, :, :3] = img[:, :, :]
    return cairo.ImageSurface.create_for_data(
        arr, cairo.FORMAT_RGB24, w, h)


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
        help='Output path of pretty visualization image (will be png)')

    args = parser.parse_args()
    img = cv2.imread(args.photo[0])

    analyzer = BoardAnalyzer()
    detected = analyzer.analyze(img)
    print("Detected: %s" % detected)
    if args.output_visualization is not None:
        surf = to_cairo_surface(img)
        ctx = cairo.Context(surf)

        # Draw grid outline with circle at origin.
        for (i, corner) in enumerate(detected.corners):
            if i == 0:
                ctx.move_to(*corner)
            else:
                ctx.line_to(*corner)
        ctx.close_path()
        ctx.set_source_rgba(0, 1, 0, 0.9)
        ctx.stroke()

        ctx.set_source_rgba(0, 1, 1, 0.9)
        ctx.arc(detected.corners[0, 0], detected.corners[0, 1], 10, 0, 2 * math.pi)
        ctx.fill()


        surf.write_to_png(args.output_visualization)

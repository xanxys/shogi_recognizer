#!/bin/python
from __future__ import print_function, division
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Analyze shogi board state in a photo""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    args = parser.parse_args()

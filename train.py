#!/bin/python2
from __future__ import print_function, division
import argparse
import cv
import cv2
import numpy as np
import theano
import theano.tensor as T
import os
import os.path


def preprocess_cell_image(img):
    """
    Convert input (80^2 RGB) to R^400 vector (20^2 grayscale image)
    in [0, 1]^400
    because number of samples are low.
    """
    assert(img.shape == (80, 80, 3))
    img_gray = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    return cv2.resize(img_gray, (20, 20), cv.CV_INTER_AREA).reshape([400]) / 255


def setup_logistic_regression():
    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.fmatrix('x')
    y = T.lvector('y')

    # allocate shared variables model params
    b = theano.shared(np.zeros((2,)), name='b')
    W = theano.shared(np.zeros((400, 2)), name='W')

    # symbolic expression for computing the matrix of class-membership probabilities
    # Where:
    # W is a matrix where column-k represent the separation hyper plain for class-k
    # x is a matrix where row-j  represents input training sample-j
    # b is a vector where element-k represent the free parameter of hyper plain-k
    p_y_given_x = T.nnet.softmax(T.dot(x, W) + b)

    # compiled Theano function that returns the vector of class-membership
    # probabilities
    get_p_y_given_x = theano.function(inputs=[x], outputs=p_y_given_x)

    # print the probability of some example represented by x_value
    # x_value is not a symbolic variable but a numpy array describing the
    # datapoint
    for i in range(2):
        print('Probability that x is of class %i is %f' % (i, get_p_y_given_x(x_value)[i]))

    # symbolic description of how to compute prediction as class whose probability
    # is maximal
    y_pred = T.argmax(p_y_given_x, axis=1)

    # compiled theano function that returns this value
    classify = theano.function(inputs=[x], outputs=y_pred)


def load_all_samples_in(dir_path):
    """
    Return [N, 400] vector (N: number of samples)
    """
    samples = []
    for path in os.listdir(dir_path):
        img_cell = cv2.imread(os.path.join(dir_path, path))
        samples.append(preprocess_cell_image(img_cell))
    return np.array(samples)

if __name__ == '__main__':
    samples_occupied = load_all_samples_in('derived/cell-occupied')
    cv2.imwrite('debug/occupied.png', samples_occupied * 255)

    samples_empty = load_all_samples_in('derived/cell-empty')
    cv2.imwrite('debug/empty.png', samples_empty * 255)

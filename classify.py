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
import random
import time
import json
import bz2


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=np.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=np.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def dump_parameters(self, path):
        """
        Dump model parameters to path in .json.bz2 format.
        Although it's somewhat human readable, format
        will not be stable.
        """
        params_w = self.W.eval()
        params_b = self.b.eval()
        blob = {
            "comment": "logistic regression, 20x20 image",
            "categories": ["occupied", "empty"],
            "W": map(list, params_w),
            "b": list(params_b)
        }
        with bz2.BZ2File(path, 'w') as bz2_stream:
            json.dump(blob, bz2_stream, separators=(',', ':'))

    def load_parameters(self, path):
        """
        Load model parameters written by dump_parameters.
        """
        blob = json.load(bz2.BZ2File(path, 'r'))
        np.array(blob["W"])
        np.array(blob["b"])

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def preprocess_cell_image(img):
    """
    Convert input (80^2 RGB) to R^400 vector (20^2 grayscale image)
    in [0, 1]^400
    because number of samples are low.
    """
    assert(img.shape == (80, 80, 3))
    img_small = cv2.resize(
        cv2.cvtColor(img, cv.CV_BGR2GRAY), (20, 20), cv.CV_INTER_AREA)
    return img_small.reshape([400]) / 255


def load_all_samples_in(dir_path, var_rotate=False):
    """
    Return [N, 400] vector (N: number of samples)
    """
    samples = []
    for path in os.listdir(dir_path):
        img_cell = cv2.imread(os.path.join(dir_path, path))
        # Create variants.
        imgs = []
        if var_rotate:
            imgs = [
                img_cell,
                img_cell[::-1, ::-1],
                img_cell.transpose([1, 0, 2])[::-1],
                img_cell.transpose([1, 0, 2])[:, ::-1]]
        else:
            imgs = [img_cell]
        for img in imgs:
            samples.append(preprocess_cell_image(img))
    return np.array(samples)


def load_data():
    """
    Loads the dataset
    """
    print('Loading images')
    samples = []
    labels = []
    for (category, name) in [(0, "occupied"), (1, "empty")]:
        dataset_path = 'derived/cell-%s' % name
        samples_cat = load_all_samples_in(dataset_path, var_rotate=True)
        cv2.imwrite('debug/%s.png' % name, samples_cat * 255)
        labels_cat = np.ones([len(samples_cat)]) * category
        samples.append(samples_cat)
        labels.append(labels_cat)
    # shuffle data
    samples = np.vstack(samples)
    labels = np.hstack(labels)
    n = len(samples)
    assert(len(labels) == n)
    ixs = range(n)
    random.shuffle(ixs)
    samples = np.array([samples[ix] for ix in ixs])
    labels = np.array([labels[ix] for ix in ixs])

    # split 3:1:1 (train, validation, test)
    x_train = samples[:int(n * 0.6)]
    y_train = labels[:int(n * 0.6)]
    x_validate = samples[int(n * 0.6):int(n * 0.8)]
    y_validate = labels[int(n * 0.6):int(n * 0.8)]
    x_test = samples[int(n * 0.8):]
    y_test = labels[int(n * 0.8):]

    print('Dataset size: all=%d : train=%d validate=%d test=%d' % (
        n, len(x_train), len(x_validate), len(x_test)))

    def shared_dataset(data_x, data_y, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        shared_x = theano.shared(
            np.asarray(data_x, dtype=theano.config.floatX),
            borrow=borrow)
        shared_y = theano.shared(
            np.asarray(data_y, dtype=theano.config.floatX),
            borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return (shared_x, T.cast(shared_y, 'int32'))

    return [
        shared_dataset(x_train, y_train),
        shared_dataset(x_validate, y_validate),
        shared_dataset(x_test, y_test)
    ]


def train_cell_classifier(dump_path, learning_rate=0.13, n_epochs=1000, batch_size=100):
    """
    Train logistic regression classifier {emtpy, occupied} with
    stochastic gradient descent.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer
    """
    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                           # [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(input=x, n_in=400, n_out=2)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    predict_model = theano.function(
        inputs=[],
        outputs=classifier.y_pred,
        givens={
            x: test_set_x})

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = np.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                    (epoch, minibatch_index + 1, n_train_batches,
                    this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of best'
                       ' model %f %%') %
                        (epoch, minibatch_index + 1, n_train_batches,
                         test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('validation(best): %f %% / test: %f %%' %
         (best_validation_loss * 100, test_score * 100))
    print('The code run for %d epochs, with %f epochs/sec' %
        (epoch, epoch / (end_time - start_time)))
    print('The code ran for %.1fs' % (end_time - start_time))

    print('Showing test set results')
    ys_pred = predict_model()
    n_all = 0
    n_fail = 0
    for (i, (tx, y, y_pred)) in enumerate(zip(test_set_x.eval(), test_set_y.eval(), ys_pred)):
        result = 'success' if y == y_pred else 'fail'
        cv2.imwrite('debug/classify-%s-%d.png' % (result, i), tx.reshape([20, 20]) * 255)
        if y != y_pred:
            n_fail += 1
        n_all += 1
    print('Failure rate: %f' % (n_fail / n_all))

    print('Writing classifier parameters to %s' % dump_path)
    classifier.dump_parameters(dump_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""
Train classifier.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--output', nargs='?', metavar='OUT_PATH', type=str, const=True,
        default='/dev/null',
        help='Parameter .json.bz2 output path')

    args = parser.parse_args()
    train_cell_classifier(args.output)

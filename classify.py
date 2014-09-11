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
import ujson
import bz2
import neuralnet


def np_to_py(arr):
    """
    Convert numpy tensor to python nested list
    for serialization.
    """
    if arr.ndim == 1:
        return list(arr)
    else:
        return map(np_to_py, arr)


class CellEmptinessClassifier(object):
    def __init__(self):
        x = T.matrix('x')
        self.x = x

        rng = np.random.RandomState(1234)
        self.mlp = neuralnet.MLP(
            input=x, n_in=400, rng=rng, n_hidden=10, n_out=2)

        def gen_cost(y):
            return (
                self.mlp.negative_log_likelihood(y) +
                self.mlp.L1 * 0 +
                self.mlp.L2_sqr * 1e-4)
        self.cost = gen_cost
        self.errors = self.mlp.errors
        self.params = self.mlp.params

        self.classify_model = theano.function(
            inputs=[x],
            outputs=[
                self.mlp.logRegressionLayer.y_pred,
                self.mlp.logRegressionLayer.p_y_given_x])

    def dump_parameters(self, path):
        ser_params = []
        for param in self.mlp.params:
            ser_params.append(np_to_py(param.eval()))

        blob = {
            "comment": "MLP, 20x20 image",
            "mlp": ser_params
        }
        with bz2.BZ2File(path, 'w') as bz2_stream:
            json.dump(blob, bz2_stream, separators=(',', ':'))

    def load_parameters(self, path):
        blob = json.load(bz2.BZ2File(path, 'r'))
        for (p, val) in zip(self.mlp.params, blob["mlp"]):
            p.set_value(np.array(val))

    def classify(self, img):
        """
        Return ("occuped" | "empty", probability of label being correct)
        """
        labels = {
            0: "occupied",
            1: "empty"
        }
        vec = preprocess_cell_image(img)
        categories, probs = self.classify_model(vec.reshape([1, 400]))
        category = categories[0]
        return (labels[category], probs[0][category])


class CellTypeClassifierUp(object):
    """
    Classify cell image with optional up-pointing piece.
    """
    def __init__(self):
        self.labels = [
            "empty",
            "FU", "KY", "KE", "GI", "KI", "KA", "HI",
            "TO", "NY", "NK", "NG",        "UM", "RY",
            "OU"
        ]
        x = T.matrix('x')
        self.x = x

        self.regression = neuralnet.LogisticRegression(
            input=x, n_in=400, n_out=len(self.labels))
        self.classify_model = theano.function(
            inputs=[x],
            outputs=[self.regression.y_pred, self.regression.p_y_given_x])

        self.cost = self.regression.negative_log_likelihood
        self.errors = self.regression.errors
        self.params = self.regression.params

    def dump_parameters(self, path):
        ser_params = []
        for param in self.mlp.params:
            ser_params.append(np_to_py(param.eval()))

        blob = {
            "comment": "logistic regression, 20x20 image",
            "regression": ser_params
        }
        with bz2.BZ2File(path, 'w') as bz2_stream:
            json.dump(blob, bz2_stream, separators=(',', ':'))

    def load_parameters(self, path):
        blob = json.load(bz2.BZ2File(path, 'r'))
        if "b" in blob and "W" in blob:
            # old format
            self.regression.b.set_value(np.array(blob["b"]))
            self.regression.W.set_value(np.array(blob["W"]))
        else:
            # new format
            for (p, val) in zip(self.regression.params, blob["regression"]):
                p.set_value(np.array(val))

    def get_label_to_category(self):
        return {label: cat for (cat, label) in enumerate(self.labels)}

    def get_category_to_label(self):
        return dict(enumerate(self.labels))

    def classify(self, img):
        """
        Return (label, prob)
        """
        vec = preprocess_cell_image(img)
        categories, probs = self.classify_model(vec.reshape([1, 400]))
        category = categories[0]
        return (self.get_category_to_label()[category], probs[0][category])


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


def produce_variant(img_seed, var_rotate=False):
    """
    return [img_variant] from img_seed
    """
    variants = [img_seed]
    if var_rotate:
        new_variants = []
        for img in variants:
            new_variants += [
                img,
                img[::-1, ::-1],
                img.transpose([1, 0, 2])[::-1],
                img.transpose([1, 0, 2])[:, ::-1]
            ]
        variants = new_variants
    return variants


def load_data():
    """
    Loads the empty vs. non-empty dataset
    """
    print('Loading images')
    samples = []
    labels = []
    dataset_path = 'derived/cells-emptiness'
    table = {
        "occupied": 0,
        "empty": 1
    }
    for path in os.listdir(dataset_path):
        photo_id, org_pos, label = os.path.splitext(path)[0].split('-')
        img_cell = cv2.imread(os.path.join(dataset_path, path))

        for img in produce_variant(img_cell, var_rotate=True):
            samples.append(preprocess_cell_image(img))
            labels.append(table[label])
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


def load_up_dataset(classifier):
    """
    Load up pieces dataset.
    """
    print('Loading images')
    samples = []
    categories = []
    table = classifier.get_label_to_category()
    print(table)
    dir_path = 'derived/cells'
    for path in os.listdir(dir_path):
        photo_id, org_pos, ptype = os.path.splitext(path)[0].split('-')
        img_cell = cv2.imread(os.path.join(dir_path, path))
        samples.append(preprocess_cell_image(img_cell))
        categories.append(table[ptype])

    # shuffle data
    samples = np.array(samples)
    categories = np.array(categories)
    n = len(samples)
    assert(len(categories) == n)
    ixs = range(n)
    random.shuffle(ixs)
    samples = np.array([samples[ix] for ix in ixs])
    categories = np.array([categories[ix] for ix in ixs])

    # split 3:1:1 (train, validation, test)
    x_train = samples[:int(n * 0.6)]
    y_train = categories[:int(n * 0.6)]
    x_validate = samples[int(n * 0.6):int(n * 0.8)]
    y_validate = categories[int(n * 0.6):int(n * 0.8)]
    x_test = samples[int(n * 0.8):]
    y_test = categories[int(n * 0.8):]

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


def train_sgd(
        datasets, model, learning_rate=0.13, n_epochs=1000, batch_size=100):
    """
    Use stochastic gradient descent to train model.
    model must expose following theano function/variables:
    * x
    * cost(y)
    * errors(y)
    * params (list of theano variables)

    datasets: (train_set, valid_set, test_set)
    """
    assert(len(datasets) == 3)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = model.x
    y = T.ivector('y')  # the labels are presented as 1D vector of
                           # [int] labels

    cost = model.cost(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=model.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function(
        inputs=[index],
        outputs=model.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    # compute the gradient of cost with respect to theta = (W,b)
    g_params = []
    for param in model.params:
        g_params.append(T.grad(cost=cost, wrt=param))

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [
        (param, param - learning_rate * g_param) for
        (param, g_param) in zip(model.params, g_params)]

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

    print('Training model')
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            train_model(minibatch_index)
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

    duration = end_time - start_time
    print('validation(best): %f %% | test: %f %%' % (best_validation_loss * 100, test_score * 100))
    print('%d epochs | %.1fsec | %fepoch/sec' % (epoch, duration, epoch / duration))


def command_train_cell_classifier(dump_path):
    """
    Train logistic regression classifier {emtpy, occupied} with
    stochastic gradient descent.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer
    """
    ct_classifier = CellEmptinessClassifier()
    datasets = load_data()

    train_sgd(datasets, ct_classifier, learning_rate=0.05, batch_size=20, n_epochs=2000)

    test_set_x, test_set_y = datasets[2]
    predict_model = theano.function(
        inputs=[],
        outputs=ct_classifier.mlp.logRegressionLayer.y_pred,
        givens={
            ct_classifier.x: test_set_x})

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
    ct_classifier.dump_parameters(dump_path)


def command_train_types_up_classifier(param_path, learning_rate=0.13, n_epochs=1000, batch_size=100):
    print('Training upright cell type classifier (MLP)')
    ct_classifier = CellTypeClassifierUp()
    datasets = load_up_dataset(ct_classifier)

    train_sgd(datasets, ct_classifier)

    test_set_x, test_set_y = datasets[2]
    predict_model = theano.function(
        inputs=[],
        outputs=ct_classifier.regression.y_pred,
        givens={
            ct_classifier.x: test_set_x})


    print('Showing test set results')
    ys_pred = predict_model()
    n_all = 0
    n_fail = 0
    for (i, (tx, y, y_pred)) in enumerate(zip(test_set_x.eval(), test_set_y.eval(), ys_pred)):
        label_pred = ct_classifier.get_category_to_label()[y_pred]
        result = 'success' if y == y_pred else 'fail'
        cv2.imwrite('debug/classify-%s-%s-%d.png' % (result, label_pred, i), tx.reshape([20, 20]) * 255)
        if y != y_pred:
            n_fail += 1
        n_all += 1
    print('Failure rate: %f' % (n_fail / n_all))

    print('Writing classifier parameters to %s' % param_path)
    ct_classifier.dump_parameters(param_path)


def command_classify_images(param_path, img_paths):
    classifier = CellEmptinessClassifier()
    classifier.load_parameters(param_path)
    for img_path in img_paths:
        img = cv2.imread(img_path)
        label, prob = classifier.classify(img)
        print('%s: %s with p=%f' % (img_path, label, prob))


def command_classify_images_up(param_path, img_paths):
    classifier = CellTypeClassifierUp()
    classifier.load_parameters(param_path)
    for img_path in img_paths:
        img = cv2.imread(img_path)
        label, prob = classifier.classify(img)
        print('%s: %s with p=%f' % (img_path, label, prob))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""
Train classifier.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--output', nargs='?', metavar='OUT_PATH', type=str, const=True,
        default='/dev/null',
        help='Parameter .json.bz2 output path')
    parser.add_argument(
        '--input', nargs='?', metavar='IN_PATH', type=str, const=True,
        default=None,
        help='Parameter .json.bz2 input path')

    parser.add_argument(
        '--classify-emptiness', nargs='+', metavar='IMG_PATH', type=str,
        default=None,
        help='Classify cell images')
    parser.add_argument(
        '--classify-types-up', nargs='+', metavar='IMG_PATH', type=str,
        default=None,
        help='Classify upright image types.')

    parser.add_argument(
        '--train-emptiness', action='store_true',
        help='Train any direction empty vs occupied cell classifier (2 categories)')
    parser.add_argument(
        '--train-types-up', action='store_true',
        help='Train upright piece type classifier (15 categories)')

    args = parser.parse_args()

    if args.classify_emptiness is not None:
        if args.input is None:
            raise RuntimeError("Specify --input to classify")
        command_classify_images(args.input, args.classify_emptiness)
    elif args.classify_types_up is not None:
        if args.input is None:
            raise RuntimeError("Specify --input to classify")
        command_classify_images_up(args.input, args.classify_types_up)
    elif args.train_types_up:
        command_train_types_up_classifier(args.output)
    elif args.train_emptiness:
        command_train_cell_classifier(args.output)
    else:
        print('Specify model to train with --train-... argument')

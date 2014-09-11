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


class CellEmptinessClassifier(object):
    def __init__(self):
        x = T.matrix('x')
        self.x = x

        self.regression = LogisticRegression(input=x, n_in=400, n_out=2)
        self.negative_log_likelihood = self.regression.negative_log_likelihood
        self.errors = self.regression.errors
        self.params = self.regression.params

        self.classify_model = theano.function(
            inputs=[x],
            outputs=[self.regression.y_pred, self.regression.p_y_given_x])

    def dump_parameters(self, path):
        self.regression.dump_parameters(path)

    def load_parameters(self, path):
        self.regression.load_parameters(path)

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

        self.regression = LogisticRegression(input=x, n_in=400, n_out=len(self.labels))
        self.classify_model = theano.function(
            inputs=[x],
            outputs=[self.regression.y_pred, self.regression.p_y_given_x])

        self.negative_log_likelihood = self.regression.negative_log_likelihood
        self.errors = self.regression.errors
        self.params = self.regression.params

    def load_parameters(self, path):
        self.regression.load_parameters(path)

    def dump_parameters(self, path):
        self.regression.dump_parameters(path)

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


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(rng.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]


class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(rng=rng, input=input,
                                       n_in=n_in, n_out=n_hidden,
                                       activation=T.tanh)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.logRegressionLayer.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() + (self.logRegressionLayer.W ** 2).sum()

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params


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
        self.W.set_value(np.array(blob["W"]))
        self.b.set_value(np.array(blob["b"]))

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
    Loads the empty vs. non-empty dataset
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


def train_sgd(datasets, model, learning_rate=0.13, n_epochs=1000, batch_size=100):
    """
    Use stochastic gradient descent to train model.
    model must expose following theano function/variables:
    * x
    * negative_log_likelihood
    * errors
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

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = model.negative_log_likelihood(y)

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


    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
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

import numpy
from non_linear import *


class UnigramLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation=relu, use_bias=False, sum_out=True):
        """
        Allocate a UnigramLayer with shared variable internal parameters.
        """

        self.input = input
        self.activation = activation
        self.use_bias = use_bias
        # initialize weights with random weights
        if "relu" in activation.func_name:
            W_values = numpy.asarray(0.01 * rng.standard_normal(size=(n_in, n_out)), dtype=theano.config.floatX)
        else:
            W_values = numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (n_in + n_out)), high=numpy.sqrt(6. / (n_in + n_out)),
                                                 size=(n_in, n_out)), dtype=theano.config.floatX)

        self.W = theano.shared(W_values, borrow=True, name="W_cov")

        b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b_cov')

        lin_out = T.dot(input, self.W) + self.b if use_bias else T.dot(input, self.W)
        unigram_sum = T.sum(lin_out, axis=1)
        self.output = activation(unigram_sum) if sum_out else activation(lin_out)

        self.params = [self.W, self.b] if use_bias else [self.W]


class BigramLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation=tanh, use_bias=False, sum_out=True):
        """
        Allocate a BigramLayer with shared variable internal parameters.
        """

        self.input = input
        self.activation = activation
        self.use_bias = use_bias
        if "relu" in activation.func_name:
            W_values = numpy.asarray(0.01 * rng.standard_normal(size=(n_in, n_out)), dtype=theano.config.floatX)
        else:
            W_values = numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (n_in + n_out)), high=numpy.sqrt(6. / (n_in + n_out)),
                                                 size=(n_in, n_out)), dtype=theano.config.floatX)

        self.Tr = theano.shared(W_values, borrow=True, name="Tr")
        self.Tl = theano.shared(W_values, borrow=True, name="Tl")

        b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b_cov')

        left = T.dot(input, self.Tl)[:, :-1]
        right = T.dot(input, self.Tr)[:, 1:]

        lin_out = left + right + self.b if use_bias else left + right
        bigram_sum = T.sum(lin_out / 2, axis=1)

        self.output = activation(bigram_sum) if sum_out else activation(lin_out)
        self.params = [self.Tr, self.Tl, self.b] if use_bias else [self.Tr, self.Tl]


class TrigramLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation=tanh, use_bias=False, sum_out=True):
        """
        Allocate a BigramLayer with shared variable internal parameters.
        """

        self.input = input
        self.activation = activation
        self.use_bias = use_bias
        if "relu" in activation.func_name:
            W_values = numpy.asarray(0.01 * rng.standard_normal(size=(n_in, n_out)), dtype=theano.config.floatX)
        else:
            W_values = numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (n_in + n_out)), high=numpy.sqrt(6. / (n_in + n_out)),
                                                 size=(n_in, n_out)), dtype=theano.config.floatX)

        self.T1 = theano.shared(W_values, borrow=True, name="T1")
        self.T2 = theano.shared(W_values, borrow=True, name="T2")
        self.T3 = theano.shared(W_values, borrow=True, name="T3")

        b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b_cov')

        left = T.dot(input, self.T1)[:, :-1]
        center = T.dot(input, self.T2)[:, :-1]
        right = T.dot(input, self.T3)[:, 1:]

        lin_out = left + center + right + self.b if use_bias else left + center + right
        trigram_sum = T.sum(lin_out, axis=1)

        self.output = activation(trigram_sum) if sum_out else activation(lin_out)
        self.params = [self.T1, self.T2, self.T3, self.b] if use_bias else [self.T1, self.T2, self.T3]

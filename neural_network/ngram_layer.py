import numpy
import theano
from non_linear import *
from initialization import get_W_values


def get_input_info(input_shape, sum_out, ngram):
    assert len(input_shape) == 2
    assert ngram <= 3
    fan_in = input_shape[0] * input_shape[1]
    n_in = n_out = input_shape[1]
    if sum_out:
        fan_out = input_shape[1]
    else:
        fan_out = (input_shape[0] - ngram + 1) * input_shape[1]
    return n_in, n_out, fan_in, fan_out


class UnigramLayer(object):
    def __init__(self, rng, input, input_shape, activation=relu, use_bias=False, sum_out=True):
        """
        Allocate a UnigramLayer with shared variable internal parameters.
        """
        self.input = input
        self.activation = activation
        self.use_bias = use_bias

        # initialize weights with random weights
        n_in, n_out, fan_in, fan_out = get_input_info(input_shape, sum_out=sum_out, ngram=1)
        W_values = get_W_values(rng=rng, activation=activation, fan_in=fan_in, fan_out=fan_out, n_in=n_in, n_out=n_out,
                                n_kernels=0)

        self.W = theano.shared(W_values, borrow=True, name="W_cov")

        b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b_cov')

        lin_out = T.dot(input, self.W) + self.b if use_bias else T.dot(input, self.W)
        activation_out = activation(lin_out)
        unigram_sum = T.sum(activation_out, axis=1)
        self.output = unigram_sum if sum_out else activation_out

        self.params = [self.W, self.b] if use_bias else [self.W]


class BigramLayer(object):
    def __init__(self, rng, input, input_shape, activation=tanh, use_bias=False, sum_out=True):
        """
        Allocate a BigramLayer with shared variable internal parameters.
        """

        self.input = input
        self.activation = activation
        self.use_bias = use_bias

        # initialize weights with random weights
        n_in, n_out, fan_in, fan_out = get_input_info(input_shape, sum_out=sum_out, ngram=2)
        W_values = get_W_values(rng=rng, activation=activation, fan_in=fan_in, fan_out=fan_out, n_in=n_in, n_out=n_out,
                                n_kernels=0)

        self.Tr = theano.shared(W_values, borrow=True, name="Tr")
        self.Tl = theano.shared(W_values, borrow=True, name="Tl")

        b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b_cov')

        left = T.dot(input, self.Tl)[:, :-1]
        right = T.dot(input, self.Tr)[:, 1:]

        lin_out = left + right + self.b if use_bias else left + right
        activation_out = activation(lin_out)
        bigram_sum = T.sum(activation_out, axis=1)

        self.output = bigram_sum if sum_out else activation_out
        self.params = [self.Tr, self.Tl, self.b] if use_bias else [self.Tr, self.Tl]


class TrigramLayer(object):
    def __init__(self, rng, input, input_shape, activation=tanh, use_bias=False, sum_out=True):
        """
        Allocate a BigramLayer with shared variable internal parameters.
        """

        self.input = input
        self.activation = activation
        self.use_bias = use_bias

        # initialize weights with random weights
        n_in, n_out, fan_in, fan_out = get_input_info(input_shape, sum_out=sum_out, ngram=3)
        W_values = get_W_values(rng=rng, activation=activation, fan_in=fan_in, fan_out=fan_out, n_in=n_in, n_out=n_out,
                                n_kernels=0)

        self.T1 = theano.shared(W_values, borrow=True, name="T1")
        self.T2 = theano.shared(W_values, borrow=True, name="T2")
        self.T3 = theano.shared(W_values, borrow=True, name="T3")

        b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b_cov')

        left = T.dot(input, self.T1)[:, :-2]
        center = T.dot(input, self.T2)[:, 1:-1]
        right = T.dot(input, self.T3)[:, 2:]

        lin_out = left + center + right + self.b if use_bias else left + center + right
        activation_out = activation(lin_out)
        trigram_sum = T.sum(activation_out, axis=1)

        self.output = trigram_sum if sum_out else activation_out
        self.params = [self.T1, self.T2, self.T3, self.b] if use_bias else [self.T1, self.T2, self.T3]


class MuiltiUnigramLayer(object):
    def __init__(self, rng, input, input_shape, activation=relu, n_kernels=4, sum_out=True, mean=True, concat_out=False):
        """
        Allocate a UnigramLayer with shared variable internal parameters.
        """

        self.input = input
        self.activation = activation

        # initialize weights with random weights
        n_in, n_out, fan_in, fan_out = get_input_info(input_shape, sum_out=sum_out, ngram=1)
        W_values = get_W_values(rng=rng, activation=activation, fan_in=fan_in, fan_out=fan_out, n_in=n_in, n_out=n_out,
                                n_kernels=n_kernels)
        self.W = theano.shared(W_values, borrow=True, name="W_cov")

        cov_out = T.dot(input, self.W)
        activation_out = activation(cov_out)

        if concat_out:
            uigram_sum = T.sum(activation_out, axis=1)
            self.output = uigram_sum.flatten(2)
        else:
            pool_out = T.mean(activation_out, axis=2) if mean else T.max(activation_out, axis=2)
            uigram_sum = T.sum(pool_out, axis=1)
            self.output = uigram_sum if sum_out else pool_out
        self.params = [self.W]


class MultiBigramLayer(object):
    def __init__(self, rng, input, input_shape, activation=tanh, n_kernels=4, mean=True, sum_out=True, concat_out=False):
        """
        Allocate a BigramLayer with shared variable internal parameters.
        """

        self.input = input
        self.activation = activation

        # initialize weights with random weights
        n_in, n_out, fan_in, fan_out = get_input_info(input_shape, sum_out=sum_out, ngram=2)
        W_values = get_W_values(rng=rng, activation=activation, fan_in=fan_in, fan_out=fan_out, n_in=n_in, n_out=n_out,
                                n_kernels=n_kernels)

        self.Tr = theano.shared(W_values, borrow=True, name="Tr")
        self.Tl = theano.shared(W_values, borrow=True, name="Tl")

        left = T.dot(input, self.Tl)[:, :-1]
        right = T.dot(input, self.Tr)[:, 1:]

        cov_out = left + right
        activation_out = activation(cov_out)

        # concatenate the output of each kernel
        if concat_out:
            bigram_sum = T.sum(activation_out, axis=1)
            self.output = bigram_sum.flatten(2)
        else:
            pool_out = T.mean(activation_out, axis=2) if mean else T.max(activation_out, axis=2)
            bigram_sum = T.sum(pool_out, axis=1)
            self.output = bigram_sum if sum_out else pool_out

        self.params = [self.Tr, self.Tl]


class MultiTrigramLayer(object):
    def __init__(self, rng, input, input_shape, activation=tanh, n_kernels=4, mean=True, sum_out=True, concat_out=False):
        """
        Allocate a BigramLayer with shared variable internal parameters.
        """

        self.input = input
        self.activation = activation

        # initialize weights with random weights
        n_in, n_out, fan_in, fan_out = get_input_info(input_shape, sum_out=sum_out, ngram=3)
        W_values = get_W_values(rng=rng, activation=activation, fan_in=fan_in, fan_out=fan_out, n_in=n_in, n_out=n_out,
                                n_kernels=n_kernels)

        self.T1 = theano.shared(W_values, borrow=True, name="T1")
        self.T2 = theano.shared(W_values, borrow=True, name="T2")
        self.T3 = theano.shared(W_values, borrow=True, name="T3")

        left = T.dot(input, self.T1)[:, :-2]
        center = T.dot(input, self.T2)[:, 1:-1]
        right = T.dot(input, self.T3)[:, 2:]

        cov_out = left + center + right
        activation_out = activation(cov_out)

        # concatenate the output of each kernel
        if concat_out:
            trigram_sum = T.sum(activation_out, axis=1)
            self.output = trigram_sum.flatten(2)
        else:
            pool_out = T.mean(activation_out, axis=2) if mean else T.max(activation_out, axis=2)
            trigram_sum = T.sum(pool_out, axis=1)
            self.output = trigram_sum if sum_out else pool_out

        self.params = [self.T1, self.T2, self.T3]

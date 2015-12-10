import numpy
import theano
from non_linear import *
from initialization import get_W_values
from helper import get_input_info


class MuiltiUnigramLayer(object):
    def __init__(self, rng, input, input_shape, activation=relu, n_kernels=4, n_out=300,
                 sum_out=True, mean=True, concat_out=False):
        """
        Allocate a UnigramLayer with shared variable internal parameters.
        """

        self.input = input
        self.activation = activation

        # initialize weights with random weights
        n_in, n_out, fan_in, fan_out = get_input_info(input_shape, sum_out=sum_out, ngram=1, n_out=n_out)
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
    def __init__(self, rng, input, input_shape, activation=tanh, n_kernels=4, n_out=300,
                 mean=True, sum_out=True, concat_out=False, skip_gram=False):
        """
        Allocate a BigramLayer with shared variable internal parameters.
        """

        self.input = input
        self.activation = activation

        # initialize weights with random weights
        n_in, n_out, fan_in, fan_out = get_input_info(input_shape, sum_out=sum_out, ngram=2, n_out=n_out)
        W_values = get_W_values(rng=rng, activation=activation, fan_in=fan_in, fan_out=fan_out, n_in=n_in, n_out=n_out,
                                n_kernels=n_kernels)

        self.Tr = theano.shared(W_values, borrow=True, name="Tr")
        self.Tl = theano.shared(W_values, borrow=True, name="Tl")

        offset = 2 if skip_gram else 1
        left = T.dot(input, self.Tl)[:, :-offset]
        right = T.dot(input, self.Tr)[:, offset:]

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
    def __init__(self, rng, input, input_shape, activation=tanh, n_kernels=4, n_out=300,
                 mean=True, sum_out=True, concat_out=False, skip_gram=False):
        """
        Allocate a BigramLayer with shared variable internal parameters.
        """

        self.input = input
        self.activation = activation

        # initialize weights with random weights
        n_in, n_out, fan_in, fan_out = get_input_info(input_shape, sum_out=sum_out, ngram=3, n_out=n_out)
        W_values = get_W_values(rng=rng, activation=activation, fan_in=fan_in, fan_out=fan_out, n_in=n_in, n_out=n_out,
                                n_kernels=n_kernels)

        self.T1 = theano.shared(W_values, borrow=True, name="T1")
        self.T2 = theano.shared(W_values, borrow=True, name="T2")
        self.T3 = theano.shared(W_values, borrow=True, name="T3")

        offset = 4 if skip_gram else 2
        left = T.dot(input, self.T1)[:, : -offset]
        center = T.dot(input, self.T2)[:, offset / 2: -offset / 2]
        right = T.dot(input, self.T3)[:, offset:]

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

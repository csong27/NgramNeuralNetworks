import numpy
from non_linear import *


class UnigramLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation=relu, use_bias=False):
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
        if use_bias:
            self.output = activation(T.sum(T.dot(input, self.W) + self.b, axis=1))
        else:
            self.output = activation(T.sum(T.dot(input, self.W), axis=1))

        self.params = [self.W, self.b] if use_bias else [self.W]


class BigramLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation=tanh, use_bias=False):
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

        bigram_sum = T.sum(left + right, axis=1)

        self.output = activation(bigram_sum + self.b) if use_bias else activation(bigram_sum)
        self.params = [self.Tr, self.Tl, self.b] if use_bias else [self.Tr, self.Tl]

    def convolutional_bigram(self):
        def inner_loop(i):
            results, _ = theano.scan(
                lambda t_0, t_p1, prior_result, Tl, Tr: prior_result + self.activation(T.dot(Tl, t_0) + T.dot(Tr, t_p1)),
                sequences=dict(input=i, taps=[0, 1]),
                outputs_info=T.zeros_like(self.b, dtype='float64'),
                non_sequences=[self.Tl, self.Tr],
                )
            if self.use_bias:
                return [results[-1] + self.b]
            else:
                return [results[-1]]
        results, _ = theano.scan(inner_loop, sequences=self.input)

        raise DeprecationWarning
import theano
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from non_linear import *


srng = RandomStreams()


def dropout(X, p=0.):
    if p != 0:
        retain_prob = 1 - p
        X = X / retain_prob * srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
    return X


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)


def shared0s(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape), dtype=dtype, name=name)


def orthogonal(shape, scale=1.1):
    """ benanne lasagne ortho init (faster than qr approach)"""
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v # pick the one with the correct shape
    q = q.reshape(shape)
    return sharedX(scale * q[:shape[0], :shape[1]])


class LSTMLayer(object):

    def __init__(self, size=256, activation=tanh, gate_activation=steeper_sigmoid, init=orthogonal,
                 truncate_gradient=-1, seq_output=False, p_drop=0., weights=None):
        self.settings = locals()
        del self.settings['self']
        self.activation_str = activation
        self.activation = activation
        self.gate_activation = gate_activation
        self.init = init
        self.size = size
        self.truncate_gradient = truncate_gradient
        self.seq_output = seq_output
        self.p_drop = p_drop
        self.weights = weights

    def connect(self, l_in, n_in):
        self.l_in = l_in
        self.n_in = n_in

        self.w_i = self.init((self.n_in, self.size))
        self.w_f = self.init((self.n_in, self.size))
        self.w_o = self.init((self.n_in, self.size))
        self.w_c = self.init((self.n_in, self.size))

        self.b_i = shared0s((self.size))
        self.b_f = shared0s((self.size))
        self.b_o = shared0s((self.size))
        self.b_c = shared0s((self.size))

        self.u_i = self.init((self.size, self.size))
        self.u_f = self.init((self.size, self.size))
        self.u_o = self.init((self.size, self.size))
        self.u_c = self.init((self.size, self.size))

        self.params = [self.w_i, self.w_f, self.w_o, self.w_c, 
            self.u_i, self.u_f, self.u_o, self.u_c,  
            self.b_i, self.b_f, self.b_o, self.b_c]

        if self.weights is not None:
            for param, weight in zip(self.params, self.weights):
                param.set_value(floatX(weight))    

    def step(self, xi_t, xf_t, xo_t, xc_t, h_tm1, c_tm1, u_i, u_f, u_o, u_c):
        i_t = self.gate_activation(xi_t + T.dot(h_tm1, u_i))
        f_t = self.gate_activation(xf_t + T.dot(h_tm1, u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + T.dot(h_tm1, u_c))
        o_t = self.gate_activation(xo_t + T.dot(h_tm1, u_o))
        h_t = o_t * self.activation(c_t)
        return h_t, c_t

    def output(self, dropout_active=False):
        X = self.l_in
        if self.p_drop > 0. and dropout_active:
            X = dropout(X, self.p_drop)
        x_i = T.dot(X, self.w_i) + self.b_i
        x_f = T.dot(X, self.w_f) + self.b_f
        x_o = T.dot(X, self.w_o) + self.b_o
        x_c = T.dot(X, self.w_c) + self.b_c
        [out, cells], _ = theano.scan(
            self.step,
            sequences=[x_i, x_f, x_o, x_c], 
            outputs_info=[T.alloc(0., X.shape[1], self.size), T.alloc(0., X.shape[1], self.size)], 
            non_sequences=[self.u_i, self.u_f, self.u_o, self.u_c],
            truncate_gradient=self.truncate_gradient
        )
        if self.seq_output:
            return out
        else:
            return out[-1]

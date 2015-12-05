import theano
import numpy as np
from theano.tensor.extra_ops import repeat
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from non_linear import *
from update_rules import shared0s, sharedX, floatX
srng = RandomStreams()


def dropout(X, p=0.):
    if p != 0:
        retain_prob = 1 - p
        X = X / retain_prob * srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
    return X


def orthogonal(shape, scale=1.1, name=None):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v  # pick the one with the correct shape
    q = q.reshape(shape)
    return sharedX(scale * q[:shape[0], :shape[1]], name=name)


class LSTM(object):
    def __init__(self, input, n_in, n_out=256, activation=tanh, gate_activation=steeper_sigmoid, init=orthogonal,
                 truncate_gradient=-1, seq_output=False, p_drop=0., weights=None, mask=None):
        self.settings = locals()
        del self.settings['self']
        self.input = input
        self.activation_str = activation
        self.activation = activation
        self.gate_activation = gate_activation
        self.init = init
        self.n_out = n_out
        self.truncate_gradient = truncate_gradient
        self.seq_output = seq_output
        self.p_drop = p_drop
        self.weights = weights
        self.n_in = n_in
        self.mask = mask
        self.connect()

    def connect(self):
        self.w_i = self.init((self.n_in, self.n_out), name="w_i")
        self.w_f = self.init((self.n_in, self.n_out), name="w_f")
        self.w_o = self.init((self.n_in, self.n_out), name="w_o")
        self.w_c = self.init((self.n_in, self.n_out), name="w_c")

        self.b_i = shared0s(self.n_out, name="b_i")
        self.b_f = shared0s(self.n_out, name="b_f")
        self.b_o = shared0s(self.n_out, name="b_o")
        self.b_c = shared0s(self.n_out, name="b_c")

        self.u_i = self.init((self.n_out, self.n_out), name="u_i")
        self.u_f = self.init((self.n_out, self.n_out), name="u_f")
        self.u_o = self.init((self.n_out, self.n_out), name="u_o")
        self.u_c = self.init((self.n_out, self.n_out), name="u_c")

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

    def step_masked(self, mask, xi_t, xf_t, xo_t, xc_t, h_tm1, c_tm1, u_i, u_f, u_o, u_c):
        i_t = self.gate_activation(xi_t + T.dot(h_tm1, u_i))
        f_t = self.gate_activation(xf_t + T.dot(h_tm1, u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + T.dot(h_tm1, u_c))
        o_t = self.gate_activation(xo_t + T.dot(h_tm1, u_o))
        h_t = o_t * self.activation(c_t)
        if mask is not None:
            if h_t.ndim == 2 and mask.ndim == 1:
                mask = mask.dimshuffle(0, 'x')
            h_t = mask * h_t + (1 - mask) * h_tm1
            c_t = mask * c_t + (1 - mask) * c_tm1
        return h_t, c_t

    def output(self, dropout_active=False, pool=True):
        X = self.input
        if self.p_drop > 0. and dropout_active:
            X = dropout(X, self.p_drop)
        x_i = T.dot(X, self.w_i) + self.b_i
        x_f = T.dot(X, self.w_f) + self.b_f
        x_o = T.dot(X, self.w_o) + self.b_o
        x_c = T.dot(X, self.w_c) + self.b_c
        if self.mask is not None:
            seq_input = [self.mask, x_i, x_f, x_o, x_c]
            step = self.step_masked
        else:
            seq_input = [x_i, x_f, x_o, x_c]
            step = self.step
        [out, _], _ = theano.scan(
            step,
            sequences=seq_input,
            outputs_info=[T.alloc(0., X.shape[1], self.n_out), T.alloc(0., X.shape[1], self.n_out)],
            non_sequences=[self.u_i, self.u_f, self.u_o, self.u_c],
            truncate_gradient=self.truncate_gradient
        )
        if pool:
            if self.mask is not None:
                sum_1 = T.sum(out, axis=1)
                sum_2 = T.sum(self.mask, axis=1).dimshuffle(0, 'x')      # length of sentence
                return sum_1 / sum_2
            # if no mask, return naive mean
            return T.mean(out, axis=1)
        elif self.seq_output:
            return out
        else:
            return out[-1]


class GatedRecurrentUnit(object):
    def __init__(self, input, n_in, n_out=256, activation=tanh, gate_activation=steeper_sigmoid, init=orthogonal,
                 truncate_gradient=-1, seq_output=False, p_drop=0., direction='forward', weights=None, mask=None):
        self.activation = activation
        self.gate_activation = gate_activation
        self.init = init
        self.n_out = n_out
        self.truncate_gradient = truncate_gradient
        self.seq_output = seq_output
        self.p_drop = p_drop
        self.weights = weights
        self.direction = direction
        self.input = input
        self.n_in = n_in
        self.mask = mask
        self.connect()

    def connect(self):
        self.h0 = shared0s((1, self.n_out), name="h_0")

        self.w_z = self.init((self.n_in, self.n_out), name="w_z")
        self.w_r = self.init((self.n_in, self.n_out), name="w_r")

        self.u_z = self.init((self.n_out, self.n_out), name="u_z")
        self.u_r = self.init((self.n_out, self.n_out), name="u_r")

        self.b_z = shared0s(self.n_out, name="b_z")
        self.b_r = shared0s(self.n_out, name="b_r")

        if 'maxout' in self.activation.func_name:
            self.w_h = self.init((self.n_in, self.n_out*2), name="w_h")
            self.u_h = self.init((self.n_out, self.n_out*2), name="u_h")
            self.b_h = shared0s((self.n_out*2), name="b_h")
        else:
            self.w_h = self.init((self.n_in, self.n_out), name="w_h")
            self.u_h = self.init((self.n_out, self.n_out), name="u_h")
            self.b_h = shared0s(self.n_out, name="b_h")

        self.params = [self.h0,
                       self.w_z, self.w_r, self.w_h,
                       self.u_z, self.u_r, self.u_h,
                       self.b_z, self.b_r, self.b_h]

        if self.weights is not None:
            for param, weight in zip(self.params, self.weights):
                param.set_value(floatX(weight))

    def step(self, xz_t, xr_t, xh_t, h_tm1, u_z, u_r, u_h):
        z = self.gate_activation(xz_t + T.dot(h_tm1, u_z))
        r = self.gate_activation(xr_t + T.dot(h_tm1, u_r))
        h_tilda_t = self.activation(xh_t + T.dot(r * h_tm1, u_h))
        h_t = z * h_tm1 + (1 - z) * h_tilda_t

        return h_t

    def step_masked(self, mask, xz_t, xr_t, xh_t, h_tm1, u_z, u_r, u_h):
        z = self.gate_activation(xz_t + T.dot(h_tm1, u_z))
        r = self.gate_activation(xr_t + T.dot(h_tm1, u_r))
        h_tilda_t = self.activation(xh_t + T.dot(r * h_tm1, u_h))
        h_t = z * h_tm1 + (1 - z) * h_tilda_t
        if mask is not None:
            if h_t.ndim == 2 and mask.ndim == 1:
                mask = mask.dimshuffle(0, 'x')
            h_t = mask * h_t + (1 - mask) * h_tm1
        return h_t

    def output(self, dropout_active=False, pool=True):
        X = self.input
        if self.direction == 'backward':
            X = X[::-1]
        if self.p_drop > 0. and dropout_active:
            X = dropout(X, self.p_drop)
        x_z = T.dot(X, self.w_z) + self.b_z
        x_r = T.dot(X, self.w_r) + self.b_r
        x_h = T.dot(X, self.w_h) + self.b_h
        if self.mask is not None:
            seq_input = [self.mask, x_z, x_r, x_h]
            step = self.step_masked
        else:
            seq_input = [x_z, x_r, x_h]
            step = self.step
        out, _ = theano.scan(
            step,
            sequences=seq_input,
            outputs_info=[repeat(self.h0, x_h.shape[1], axis=0)],
            non_sequences=[self.u_z, self.u_r, self.u_h],
            truncate_gradient=self.truncate_gradient
        )
        if pool:
            if self.mask is not None:
                sum_1 = T.sum(out, axis=1)
                sum_2 = T.sum(self.mask, axis=1).dimshuffle(0, 'x')      # length of sentence
                return sum_1 / sum_2
            return T.mean(out, axis=1)
        elif self.seq_output:
            return out
        else:
            return out[-1]

if __name__ == '__main__':
    x = T.tensor3()
    mask = T.matrix()
    layer = LSTM(input=x, n_in=3, n_out=10, seq_output=True, p_drop=0.5, mask=None)
    output = layer.output(pool=False)
    f = theano.function([x, mask], output, on_unused_input='ignore')
    print f([[[1, 2, 3], [2, 3, 4], [3, 4, 5]], [[3, 4, 5], [2, 3, 4], [1, 2, 3]]], [[1, 0, 0], [1, 1, 0]])
    print layer.params

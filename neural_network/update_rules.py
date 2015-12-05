import theano
import theano.tensor as T
import numpy as np
from collections import OrderedDict
from lasagne.updates import *


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)


def shared0s(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape), dtype=dtype, name=name)


def updates_adadelta(params, cost, rho=0.95, epsilon=1e-6, norm_lim=9, word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step = -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name != 'Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
    return updates


def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)
    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)


def clip_norm(g, c, n):
    if c > 0:
        g = T.switch(T.ge(n, c), g*c/n, g)
    return g


def clip_norms(gs, c):
    norm = T.sqrt(sum([T.sum(g**2) for g in gs]))
    return [clip_norm(g, c, norm) for g in gs]


class Regularizer(object):
    def __init__(self, l1=0., l2=0., maxnorm=0.):
        self.__dict__.update(locals())

    def max_norm(self, p, maxnorm):
        if maxnorm > 0:
            norms = T.sqrt(T.sum(T.sqr(p), axis=0))
            desired = T.clip(norms, 0, maxnorm)
            p *= (desired/ (1e-7 + norms))
        return p

    def gradient_regularize(self, p, g):
        g += p * self.l2
        g += T.sgn(p) * self.l1
        return g

    def weight_regularize(self, p):
        p = self.max_norm(p, self.maxnorm)
        return p


class Update(object):
    def __init__(self, regularizer=Regularizer(), clipnorm=0.):
        self.__dict__.update(locals())

    def get_updates(self, params, grads):
        raise NotImplementedError


class SGD(Update):
    def __init__(self, lr=0.01, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def get_updates(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p,g in zip(params,grads):
            g = self.regularizer.gradient_regularize(p, g)
            updated_p = p - self.lr * g
            updated_p = self.regularizer.weight_regularize(updated_p)
            updates.append((p, updated_p))
        return updates


class Momentum(Update):
    def __init__(self, lr=0.01, momentum=0.9, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def get_updates(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p,g in zip(params,grads):
            g = self.regularizer.gradient_regularize(p, g)
            m = theano.shared(p.get_value() * 0.)
            v = (self.momentum * m) - (self.lr * g)
            updates.append((m, v))

            updated_p = p + v
            updated_p = self.regularizer.weight_regularize(updated_p)
            updates.append((p, updated_p))
        return updates


class NAG(Update):
    def __init__(self, lr=0.01, momentum=0.9, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def get_updates(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p, g in zip(params, grads):
            g = self.regularizer.gradient_regularize(p, g)
            m = theano.shared(p.get_value() * 0.)
            v = (self.momentum * m) - (self.lr * g)
            updates.append((m, v))

            updated_p = p + self.momentum * v - self.lr * g
            updated_p = self.regularizer.weight_regularize(updated_p)
            updates.append((p, updated_p))
        return updates


class RMSprop(Update):
    def __init__(self, lr=0.001, rho=0.9, epsilon=1e-6, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def get_updates(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p,g in zip(params,grads):
            g = self.regularizer.gradient_regularize(p, g)
            acc = theano.shared(p.get_value() * 0.)
            acc_new = self.rho * acc + (1 - self.rho) * g ** 2
            updates.append((acc, acc_new))

            updated_p = p - self.lr * (g / T.sqrt(acc_new + self.epsilon))
            updated_p = self.regularizer.weight_regularize(updated_p)
            updates.append((p, updated_p))
        return updates


class Adagrad(Update):
    def __init__(self, lr=0.01, epsilon=1e-6, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def get_updates(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p, g in zip(params, grads):
            g = self.regularizer.gradient_regularize(p, g)
            acc = theano.shared(p.get_value() * 0.)
            acc_t = acc + g ** 2
            updates.append((acc, acc_t))

            p_t = p - (self.lr / T.sqrt(acc_t + self.epsilon)) * g
            p_t = self.regularizer.weight_regularize(p_t)
            updates.append((p, p_t))
        return updates


class Adadelta(Update):
    def __init__(self, lr=0.5, rho=0.95, epsilon=1e-6, *args, **kwargs):
        Update.__init__(self, *args, **kwargs)
        self.__dict__.update(locals())

    def get_updates(self, params, cost):
        updates = []
        grads = T.grad(cost, params)
        grads = clip_norms(grads, self.clipnorm)
        for p,g in zip(params,grads):
            g = self.regularizer.gradient_regularize(p, g)

            acc = theano.shared(p.get_value() * 0.)
            acc_delta = theano.shared(p.get_value() * 0.)
            acc_new = self.rho * acc + (1 - self.rho) * g ** 2
            updates.append((acc, acc_new))

            update = g * T.sqrt(acc_delta + self.epsilon) / T.sqrt(acc_new + self.epsilon)
            updated_p = p - self.lr * update
            updated_p = self.regularizer.weight_regularize(updated_p)
            updates.append((p, updated_p))

            acc_delta_new = self.rho * acc_delta + (1 - self.rho) * update ** 2
            updates.append((acc_delta, acc_delta_new))
        return updates

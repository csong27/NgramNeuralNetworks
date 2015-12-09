from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano
import numpy as np
import theano.tensor as T

srng = RandomStreams()


def dropout(X, p=0.):
    if p != 0:
        retain_prob = 1 - p
        X = X / retain_prob * srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
    return X


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def t_floatX(variable):
    return T.cast(variable, theano.config.floatX)


def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)


def shared0s(shape, dtype=theano.config.floatX, name=None):
    return sharedX(np.zeros(shape), dtype=dtype, name=name)


def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')


def l2_regularization(l2_ratio, params, cost, norm_words=True):
    for param in params:
        if (param.name == 'Words') and norm_words:
            cost += T.sum(param ** 2) * l2_ratio
        if param.ndim > 1 and (param.name != 'Words'):
            cost += T.sum(param ** 2) * l2_ratio
    return cost


import theano
import numpy as np
import theano.tensor as T
import theano.tensor.shared_randomstreams


def dropout_rows(rng, input, p):
    assert input.ndim == 2
    mask = np.random.binomial(n=1, p=p, size=input.shape)
    output = input * mask
    return output


def _dropout_from_layer(rng, layer, p):
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output


def get_fans(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out


def get_input_info(input_shape, sum_out, ngram, n_out):
    assert len(input_shape) == 2
    assert ngram <= 3
    fan_in = input_shape[0] * input_shape[1]
    n_in = input_shape[1]
    if sum_out:
        fan_out = n_out
    else:
        fan_out = (input_shape[0] - ngram + 1) * n_out
    return n_in, n_out, fan_in, fan_out


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


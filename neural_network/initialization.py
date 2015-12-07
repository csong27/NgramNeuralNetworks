import numpy as np
import theano
from helper import sharedX


def get_W_values(rng, activation, fan_in, fan_out, n_in, n_out, n_kernels, normal=True, gain=1.0, Xavier=True):
    if n_kernels != 0:
        size = (n_kernels, n_in, n_out)
    else:
        size = (n_in, n_out)
    # activation
    if "relu" in activation.func_name:
        gain = np.sqrt(2.0)
    # Xavier or He
    if Xavier:
        std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    else:
        std = gain * np.sqrt(1.0 / fan_in)
    # normal or uniform
    if normal:
        return np.asarray(rng.normal(0.0, std, size=size), dtype=theano.config.floatX)
    else:
        std *= np.sqrt(3.0)
        return np.asarray(rng.uniform(low=-std, high=std, size=size), dtype=theano.config.floatX)


def orthogonal(shape, scale=1.1, name=None):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v  # pick the one with the correct shape
    q = q.reshape(shape)
    return sharedX(scale * q[:shape[0], :shape[1]], name=name)

import theano
import theano.tensor as T


def relu(x):
    return T.maximum(0.0, x)


def LeakyReLU(x, alpha):
    return theano.tensor.switch(x > 0, x, x/alpha)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def tanh(x):
    return T.tanh(x)

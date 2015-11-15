import theano
import theano.tensor as T


def relu(x):
    return T.nnet.relu(x)


def leaky_relu(x, alpha=0.2):
    return T.nnet.relu(x, alpha=alpha)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def softplus(x):
    return T.nnet.softplus(x)


def tanh(x):
    return T.tanh(x)

import theano.tensor as T


def relu(x):
    return T.nnet.relu(x)


def leaky_relu(x, alpha=0.01):
    return T.nnet.relu(x, alpha=alpha)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def softplus(x):
    return T.nnet.softplus(x)


def tanh(x):
    return T.tanh(x)


def identity(x):
    return x


def cube(x):
    return T.power(x, 3)


def maxout(x):
    return T.maximum(x[:, 0::2], x[:, 1::2])


def conv_maxout(x):
    return T.maximum(x[:, 0::2, :, :], x[:, 1::2, :, :])


def clipped_maxout(x):
    return T.clip(T.maximum(x[:, 0::2], x[:, 1::2]), -5., 5.)


def clipped_rectify(x):
    return T.clip((x + abs(x)) / 2.0, 0., 5.)


def hard_tanh(x):
    return T.clip(x, -1. , 1.)


def steeper_sigmoid(x):
    return 1./(1. + T.exp(-3.75 * x))


def hard_sigmoid(x):
    return T.clip(x + 0.5, 0., 1.)


def elu(x):
    return T.switch(x > 0, x, T.exp(x) - 1)

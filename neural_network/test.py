import numpy
import theano
import theano.tensor as T

Tl = theano.shared(numpy.ones((3, 3)).astype(theano.config.floatX))
Tr = theano.shared(numpy.ones((3, 3)).astype(theano.config.floatX))
b = theano.shared(numpy.ones((3,), dtype=theano.config.floatX))

s = T.tensor3('s')

dot1 = T.dot(s, Tl)[:, : -1]

dot2 = T.dot(s, Tr)[:, 1:]

out = T.nnet.ultra_fast_sigmoid(T.sum(dot1 + dot2, axis=1) + b)

f = theano.function([s], out)


print f([[[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8]]])

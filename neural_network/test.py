import numpy
import theano
import theano.tensor as T

W_value = numpy.random.uniform(-1, 1, (4, 3, 3))

Tl = theano.shared(W_value.astype(theano.config.floatX))
Tr = theano.shared(W_value.astype(theano.config.floatX))

s = T.tensor3('s')

dot1 = T.dot(s, Tl)
dot2 = T.dot(s, Tr)

max1 = T.max(dot1, axis=2)

f = theano.function([s], dot1)
f1 = theano.function([s], max1)


print f([[[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]])
print f1([[[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]])

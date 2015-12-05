import numpy
import theano
import theano.tensor as T


W_value = numpy.random.normal(size=(5, 5))
print W_value
W = theano.shared(W_value.astype(theano.config.floatX))

x = T.imatrix('index')

s = W[x]

f = theano.function([x], s)

print f([[0, 1, 2], [2, 1, 0]])
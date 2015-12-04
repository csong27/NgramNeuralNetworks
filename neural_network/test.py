import numpy
import theano
import theano.tensor as T


W_value = numpy.ones((5, 3, 3))

Tl = theano.shared(W_value.astype(theano.config.floatX))
Tr = theano.shared(W_value.astype(theano.config.floatX))

i = T.iscalar('index')
s = T.itensor4('s')

dot1 = T.dot(s, Tl)
dot2 = T.dot(s, Tr)

sum = T.sum(dot1, axis=1).flatten(2)

a = [[[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
     [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]]

x = numpy.asarray(a)
print x.shape

x.reshape((x.shape[0], 1, x.shape[1], x.shape[2]))

f = theano.function([i], dot1, givens={s: x}, on_unused_input='ignore')
f1 = theano.function([i], sum, givens={s: x}, on_unused_input='ignore')

print f(0)
print f1(0)

# print f1([[[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]])

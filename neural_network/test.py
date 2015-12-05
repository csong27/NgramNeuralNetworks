import numpy
import theano
import theano.tensor as T

s = numpy.random.normal(size=(2, 5, 10))
s = numpy.asarray(s, dtype=theano.config.floatX)
m = numpy.asarray([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=theano.config.floatX)

mask = T.matrix()
sentence = T.tensor3()

sum_1 = T.sum(sentence, axis=1)
sum_2 = T.sum(mask, axis=1).dimshuffle(0, 'x')

out = sum_1 / sum_2

f = theano.function([sentence, mask], out)

print f(s, m)


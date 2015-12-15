import theano
import theano.tensor as T
import numpy as np


rng = np.random.RandomState(23455)

x = np.arange(60, dtype='float32').reshape((5, 4, 3))
print x

t = T.tensor3()
b = theano.shared(np.ones(shape=(3, ), dtype='float32'))


concat = t + b

f = theano.function([t], concat, on_unused_input='ignore')

print f(x)






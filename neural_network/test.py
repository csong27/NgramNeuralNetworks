import theano
import theano.tensor as T
import numpy as np


rng = np.random.RandomState(23455)

x = np.arange(60, dtype='float32').reshape((5, 4, 3))
print x

t = T.tensor3()
concat = T.concatenate((t, t), axis=2)

f = theano.function([t], concat, on_unused_input='ignore')

print f(x)






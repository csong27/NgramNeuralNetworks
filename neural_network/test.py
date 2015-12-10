import theano
import theano.tensor as T
import numpy as np
from helper import dropout_rows_3d

rng = np.random.RandomState(23455)

x = np.arange(60, dtype='float32').reshape((5, 4, 3))
t = T.tensor3()

f = theano.function([t], dropout_rows_3d(rng, t, 0.1), on_unused_input='ignore')

print f(x)






import theano
import numpy as np
from non_linear import *


x = np.arange(27, dtype='float32').reshape((3, 3, 3))
W = theano.shared(np.ones((3, 3)))

t = T.tensor3()

out = T.dot(t, W)
act_out = maxout(out)

z = T.cast(T.alloc(0., 3, 3), theano.config.floatX)

f = theano.function([t], z, on_unused_input='ignore')

print f(x).dtype






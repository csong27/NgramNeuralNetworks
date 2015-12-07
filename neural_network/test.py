import theano
import numpy as np
from non_linear import *


x = np.arange(27, dtype='float32').reshape((3, 3, 3))
W = theano.shared(np.ones((3, 3)))

t = T.tensor3()

out = T.dot(t, W)
act_out = maxout(out)

f = theano.function([t], [out, act_out])

print f(x)






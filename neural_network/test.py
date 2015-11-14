import numpy
import theano
import theano.tensor as T

Tl = theano.shared(numpy.ones((3, 3)).astype(theano.config.floatX))
Tr = theano.shared(numpy.ones((3, 3)).astype(theano.config.floatX))
b = theano.shared(numpy.ones((3,), dtype=theano.config.floatX))

s = T.tensor3('s')


def inner_loop(i):
    results, _ = theano.scan(
        lambda t_0, t_p1, prior_result, Tl, Tr, b: prior_result + T.tanh(T.dot(Tl, t_0) + T.dot(Tr, t_p1)),
        sequences=dict(input=i, taps=[0, 1]),
        outputs_info=T.zeros_like(b, dtype='float64'),
        non_sequences=[Tl, Tr, b],
        )
    return [results[-1]]

results, _ = theano.scan(inner_loop, sequences=s)

cost = T.sum(T.dot(results, Tr))

grad = [T.grad(cost, param) for param in [Tl, Tr]]

f = theano.function([s], results)


print f([[[1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]])

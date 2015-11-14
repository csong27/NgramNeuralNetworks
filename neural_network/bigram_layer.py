import numpy
import theano.tensor.shared_randomstreams
import theano
import theano.tensor as T


class BigramLayer(object):
    def __init__(self, rng, input, n_in, n_out, non_linear="tanh"):
        """
        Allocate a BigramLayer with shared variable internal parameters.
        """

        self.input = input
        self.non_linear = non_linear

        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (n_in + n_out))

        self.Tr = theano.shared(numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=(n_in, n_out)),
                                dtype=theano.config.floatX), borrow=True, name="Tr")
        self.Tl = theano.shared(numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=(n_in, n_out)),
                                dtype=theano.config.floatX), borrow=True, name="Tl")

        b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b_cov')
        self.output = self.convolutional_bigram()
        self.params = [self.Tr, self.Tl, self.b]

    def convolutional_bigram(self):
        def inner_loop(i):
            results, _ = theano.scan(
                lambda t_0, t_p1, prior_result, Tl, Tr, b: prior_result + T.tanh(T.dot(Tl, t_0) + T.dot(Tr, t_p1)),
                sequences=dict(input=i, taps=[0, 1]),
                outputs_info=T.zeros_like(self.b, dtype='float64'),
                non_sequences=[self.Tl, self.Tr, self.b],
                )
            return [results[-1]]

        results, _ = theano.scan(inner_loop, sequences=self.input)

        return results


def test():
    s = T.tensor3('s')
    rng = numpy.random.RandomState(1234)

    bigram_layer = BigramLayer(rng=rng, input=s, n_in=3, n_out=3)

    f = theano.function([s], bigram_layer.output)

    print f([[[1, 1, 1], [0, 0, 0]], [[1, 1, 1], [1, 1, 1]]])


if __name__ == '__main__':
    test()
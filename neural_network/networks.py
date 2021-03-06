from regular_layer import _dropout_from_layer, DropoutHiddenLayer, HiddenLayer, LogisticRegression
from ngram_layer import MuiltiUnigramLayer, MultiBigramLayer, MultiTrigramLayer
from recurrent_layer import GatedRecurrentUnit, LSTM, BidirectionalRecurrentLayer
from non_linear import *
import theano.tensor as T


class MLPDropout(object):
    """A multilayer perceptron with dropout"""
    def __init__(self, rng, input, layer_sizes, dropout_rates, activations, use_bias=True):
        # Set up all the hidden layers
        self.weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])
        self.layers = []
        self.dropout_layers = []
        self.activations = activations
        next_layer_input = input
        # first_layer = True
        # dropout the input
        next_dropout_layer_input = _dropout_from_layer(rng, input, p=dropout_rates[0])
        layer_counter = 0
        for n_in, n_out in self.weight_matrix_sizes[:-1]:
            next_dropout_layer = DropoutHiddenLayer(rng=rng,
                                                    input=next_dropout_layer_input,
                                                    activation=activations[layer_counter],
                                                    n_in=n_in, n_out=n_out, use_bias=use_bias,
                                                    dropout_rate=dropout_rates[layer_counter])
            self.dropout_layers.append(next_dropout_layer)
            next_dropout_layer_input = next_dropout_layer.output

            # Reuse the parameters from the dropout layer here, in a different
            # path through the graph.
            next_layer = HiddenLayer(rng=rng,
                                     input=next_layer_input,
                                     activation=activations[layer_counter],
                                     W=next_dropout_layer.W * (1 - dropout_rates[layer_counter]),
                                     b=next_dropout_layer.b,
                                     n_in=n_in, n_out=n_out,
                                     use_bias=use_bias)
            self.layers.append(next_layer)
            next_layer_input = next_layer.output
            #   first_layer = False
            layer_counter += 1

        # Set up the output layer
        n_in, n_out = self.weight_matrix_sizes[-1]
        dropout_output_layer = LogisticRegression(
                input=next_dropout_layer_input,
                n_in=n_in, n_out=n_out)
        self.dropout_layers.append(dropout_output_layer)

        # Again, reuse parameters in the dropout output.
        output_layer = LogisticRegression(
            input=next_layer_input,
            # scale the weight matrix W with (1-p)
            W=dropout_output_layer.W * (1 - dropout_rates[-1]),
            b=dropout_output_layer.b,
            n_in=n_in, n_out=n_out)
        self.layers.append(output_layer)

        # Use the negative log likelihood of the logistic regression layer as
        # the objective.
        self.dropout_negative_log_likelihood = self.dropout_layers[-1].negative_log_likelihood
        self.dropout_cross_entropy = self.dropout_layers[-1].cross_entropy
        self.dropout_hinge_loss = self.dropout_layers[-1].multiclass_hinge_loss

        self.dropout_errors = self.dropout_layers[-1].errors

        self.negative_log_likelihood = self.layers[-1].negative_log_likelihood
        self.cross_entropy = self.layers[-1].cross_entropy

        self.errors = self.layers[-1].errors
        self.prediction = self.layers[-1].y_pred
        # Grab all the parameters together.
        self.params = [param for layer in self.dropout_layers for param in layer.params]

    def predict(self, new_data):
        next_layer_input = new_data
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                next_layer_input = self.activations[i](T.dot(next_layer_input, layer.W) + layer.b)
            else:
                p_y_given_x = T.nnet.softmax(T.dot(next_layer_input, layer.W) + layer.b)
        y_pred = T.argmax(p_y_given_x, axis=1)
        return y_pred

    def predict_p(self, new_data):
        next_layer_input = new_data
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                next_layer_input = self.activations[i](T.dot(next_layer_input, layer.W) + layer.b)
            else:
                p_y_given_x = T.nnet.softmax(T.dot(next_layer_input, layer.W) + layer.b)
        return p_y_given_x


class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out, activation):
        self.hiddenLayer = HiddenLayer(rng=rng, input=input,
                                       n_in=n_in, n_out=n_hidden,
                                       activation=activation)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out)

        self.cross_entropy = self.logRegressionLayer.cross_entropy
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        self.multiclass_hinge_loss = self.logRegressionLayer.multiclass_hinge_loss

        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params


def get_ngram_layer(ngram, args):
    if ngram == 1:
        ngram_layer = MuiltiUnigramLayer(**args)
    elif ngram == 2:
        ngram_layer = MultiBigramLayer(**args)
    elif ngram == 3:
        ngram_layer = MultiTrigramLayer(**args)
    else:
        raise NotImplementedError('This %d gram layer is not implemented' % ngram)
    return ngram_layer


class MultiKernelNgramNetwork(object):
    def __init__(self, rng, input, input_shape, ngrams=(3, 2, 1), n_kernels=(4, 4, 4), ngram_out=(300, 200, 100),
                 mean_pool=False, activation=tanh, concat_out=False, skip_gram=False, sum_out=False, mask=None,
                 use_bias=True):
        assert len(ngrams) == len(n_kernels) == len(ngram_out)    # need to have same number of layers
        self.layers = []
        prev_out = input
        for i, ngram in enumerate(ngrams[:-1]):
            if mask is not None and ngram != 1:
                mask = mask[:, : - (ngram - 1)]
            x = prev_out
            n_out = ngram_out[i]
            args = {
                "rng": rng,
                "input": x,
                "input_shape": input_shape,
                "activation": activation,
                "mean_pool": mean_pool,
                "pool_out": True,
                "sum_out": False,
                "n_kernels": n_kernels[i],
                "n_out": n_out,
                "skip_gram": skip_gram,
                "mask": mask,
                "use_bias": use_bias
            }
            ngram_layer = get_ngram_layer(ngram=ngram, args=args)
            self.layers.append(ngram_layer)
            prev_out = ngram_layer.output
            input_shape = (input_shape[0] - ngram + 1, n_out)
        ngram = ngrams[-1]
        if mask is not None and ngram != 1:
            mask = mask[:, : - (ngram - 1)]

        n_out = ngram_out[-1]
        x = self.layers[-1].output if len(self.layers) >= 1 else input
        args = {
            "rng": rng,
            "input": x,
            "input_shape": input_shape,
            "activation": activation,
            "mean_pool": mean_pool,
            "pool_out": False,
            "sum_out": sum_out,
            "n_kernels": n_kernels[-1],
            "n_out": n_out,
            "skip_gram": skip_gram,
            "concat_out": concat_out,
            "mask": mask,
            "use_bias": use_bias
        }
        last_layer = get_ngram_layer(ngram=ngram, args=args)
        self.layers.append(last_layer)
        self.output = self.layers[-1].output
        self.params = [param for layer in self.layers for param in layer.params]


class NgramRecurrentNetwork(object):
    def __init__(self, rng, input, input_shape, ngrams=(3, 2, 1), n_kernels=(4, 4, 4), mean_pool=False, mask=None,
                 ngram_out=(300, 200, 100), ngram_activation=tanh, rec_type='lstm', rec_hidden=150, n_out=2,
                 dropout_rate=0.5, rec_activation=tanh, concat_out=False, clipping=10, mlp=False,
                 mlp_hidden=300, mlp_activation=leaky_relu, skip_gram=False):

        assert len(ngrams) == len(n_kernels) == len(ngram_out)    # need to have same number of layers
        self.layers = []
        prev_out = input
        for i, ngram in enumerate(ngrams):
            x = prev_out
            concat = concat_out and (i == len(ngrams) - 1)
            tmp_out = ngram_out[i] * n_kernels[i] if concat else ngram_out[i]
            args = {
                "rng": rng,
                "input": x,
                "input_shape": input_shape,
                "activation": ngram_activation,
                "mean_pool": mean_pool,
                "sum_out": False,
                "n_kernels": n_kernels[i],
                "n_out": tmp_out,
                "concat_out": concat
            }
            ngram_layer = get_ngram_layer(ngram=ngram, args=args)
            self.layers.append(ngram_layer)
            prev_out = ngram_layer.output
            input_shape = (input_shape[0] - ngram + 1, tmp_out)
        # get the shape of mask correct for ngram layer output
        if mask is not None:
            offset = sum(ngrams) - len(ngrams)
            if offset != 0:
                mask = mask[:, : -offset]
        # recurrent layer
        rec_input = prev_out
        if rec_type == 'lstm':
            rec_layer = LSTM(input=rec_input, n_in=input_shape[1], n_out=rec_hidden, p_drop=dropout_rate, mask=mask,
                             seq_output=False, activation=rec_activation, clipping=clipping)
        elif rec_type == 'gru':
            rec_layer = GatedRecurrentUnit(input=rec_input, n_in=input_shape[1], n_out=rec_hidden, p_drop=dropout_rate,
                                           mask=mask, seq_output=False, activation=rec_activation, clipping=clipping)
        else:
            raise NotImplementedError('This %s is not implemented' % rec_type)
        self.layers.append(rec_layer)
        rec_output = rec_layer.output(pool=True)
        if mlp:
            self.classifier = MLPDropout(
                rng=rng,
                input=rec_output,
                dropout_rates=[dropout_rate],
                activations=[mlp_activation],
                layer_sizes=[rec_hidden, mlp_hidden, n_out]
            )
        else:
            self.classifier = LogisticRegression(
                input=rec_output,
                n_in=rec_hidden,
                n_out=n_out
            )
        self.negative_log_likelihood = self.classifier.negative_log_likelihood
        self.errors = self.classifier.errors
        self.params = [param for layer in self.layers for param in layer.params] + self.classifier.params


class ReversedNgramRecurrentNetwork(object):
    def __init__(self, rng, input, input_shape, ngrams=(3, 2, 1), n_kernels=(4, 4, 4), mean_pool=False, mask=None,
                 ngram_out=(300, 200, 100), ngram_activation=tanh, rec_type='lstm', rec_hidden=150, n_out=2,
                 dropout_rate=0.5, rec_activation=tanh, mlp=True, mlp_activation=leaky_relu, mlp_hidden=300,
                 concat_out=False, clipping=10, skip_gram=False, bidirection=False, sum_out=False, use_bias=True):
        assert len(ngrams) == len(n_kernels) == len(ngram_out)    # need to have same number of layers
        # recurrent layer in the bottom
        if bidirection:
            self.rec_layer = BidirectionalRecurrentLayer(input=input, n_in=input_shape[1], n_out=rec_hidden,
                                                         p_drop=dropout_rate, mask=mask, seq_output=True,
                                                         rec_type=rec_type, concat_out=True, activation=rec_activation,
                                                         clipping=clipping)
            rec_hidden *= 2
        elif rec_type == 'lstm':
            self.rec_layer = LSTM(input=input, n_in=input_shape[1], n_out=rec_hidden, p_drop=dropout_rate, mask=mask,
                                  seq_output=True, activation=rec_activation, clipping=clipping)
        elif rec_type == 'gru':
            self.rec_layer = GatedRecurrentUnit(input=input, n_in=input_shape[1], n_out=rec_hidden, p_drop=dropout_rate,
                                                mask=mask, seq_output=True, activation=rec_activation, clipping=clipping)
        else:
            raise NotImplementedError('This %s is not implemented' % rec_type)

        rec_output = self.rec_layer.output(pool=False)
        # input shape for ngram net
        input_shape = (input_shape[0], rec_hidden)
        self.ngram_net = MultiKernelNgramNetwork(
            rng=rng,
            input=rec_output,
            input_shape=input_shape,
            ngram_out=ngram_out,
            ngrams=ngrams,
            n_kernels=n_kernels,
            activation=ngram_activation,
            mean_pool=mean_pool,
            concat_out=concat_out,
            skip_gram=skip_gram,
            mask=mask,
            sum_out=sum_out,
            use_bias=use_bias
        )
        classifier_input = self.ngram_net.output
        n_in = ngram_out[-1] * n_kernels[-1] if concat_out else ngram_out[-1]
        if mlp:
            self.classifier = MLPDropout(
                rng=rng,
                input=classifier_input,
                dropout_rates=[dropout_rate],
                activations=[mlp_activation],
                layer_sizes=[n_in, mlp_hidden, n_out],
                use_bias=use_bias
            )
        else:
            self.classifier = LogisticRegression(
                input=classifier_input,
                n_in=n_in,
                n_out=n_out
            )
        self.negative_log_likelihood = self.classifier.negative_log_likelihood
        self.errors = self.classifier.errors
        self.params = self.rec_layer.params + self.ngram_net.params + self.classifier.params


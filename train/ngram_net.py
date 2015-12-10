from neural_network import *
from train import main_loop
import numpy as np
import theano


def train_ngram_net(
        datasets,
        U,
        input_shape,
        non_static=True,
        ngrams=(2, 1),
        ngram_out=(200, 100),
        ngram_activation=tanh,
        n_epochs=25,
        concat_out=False,
        n_kernels=(4, 3),
        use_bias=False,
        mean_pool=False,
        shuffle_batch=True,
        batch_size=50,
        dropout=True,
        n_hidden=100,
        n_out=2,
        activation=relu,
        dropout_rate=0.3,
        update_rule='adadelta',
        lr_rate=0.01,
        momentum_ratio=0.9,
        l2_ratio=1e-4,
        validation_only=False,
        skip_gram=False,
        word_dropout_rate=0.5
):
    rng = np.random.RandomState(23455)

    train_x, train_y, validate_x, validate_y, test_x, test_y = datasets

    print 'size of train, validation, test set are %d, %d, %d' % (train_y.shape[0], validate_y.shape[0], test_x.shape[0])
    if word_dropout_rate > 0:
        train_x = dropout_rows(rng=rng, input=train_x, p=word_dropout_rate)

    train_x, train_y = shared_dataset((train_x, train_y))
    validate_x, validate_y = shared_dataset((validate_x, validate_y))
    test_x, test_y = shared_dataset((test_x, test_y))

    n_train_batches = train_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    print 'building network, number of mini-batches are %d...' % n_train_batches

    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')
    y = T.ivector('y')

    # word vectors
    Words = theano.shared(value=U.astype(theano.config.floatX), name="Words")
    # reset zero padding
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(U.shape[1]).astype(theano.config.floatX)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0, :], zero_vec_tensor))])

    ngram_input = Words[T.cast(x, dtype="int32")]

    ngram_net = MultiKernelNgramNetwork(rng=rng,
                                        input=ngram_input,
                                        input_shape=input_shape,
                                        ngrams=ngrams,
                                        ngram_out=ngram_out,
                                        n_kernels=n_kernels,
                                        activation=ngram_activation,
                                        mean=mean_pool,
                                        concat_out=concat_out,
                                        skip_gram=skip_gram)

    dim = ngram_out[-1]
    mlp_input = ngram_net.output
    mlp_n_in = dim * n_kernels[-1] if concat_out else dim

    if dropout:
        layer_sizes = [mlp_n_in, n_hidden, n_out]
        mlp = MLPDropout(
            rng=rng,
            input=mlp_input,
            layer_sizes=layer_sizes,
            dropout_rates=[dropout_rate],
            activations=[activation],
            use_bias=use_bias
        )
        cost = mlp.dropout_negative_log_likelihood(y)
    else:
        mlp = MLP(
            rng=rng,
            input=mlp_input,
            n_in=mlp_n_in,
            n_hidden=n_hidden,
            n_out=n_out,
            activation=activation
        )
        cost = mlp.negative_log_likelihood(y)

    params = ngram_net.params + mlp.params
    # learning word vectors as well
    if non_static:
        params += [Words]
    # L2 norm
    if l2_ratio > 0:
        cost = l2_regularization(l2_ratio, params, cost)
    grad_updates = get_grad_updates(update_rule=update_rule, cost=cost, params=params, lr_rate=lr_rate,
                                    momentum_ratio=momentum_ratio)

    # functions for training
    train_model = theano.function([index], cost, updates=grad_updates,
                                  givens={
                                      x: train_x[index * batch_size:(index + 1) * batch_size],
                                      y: train_y[index * batch_size:(index + 1) * batch_size]
                                  })
    val_model = theano.function([index], mlp.errors(y), on_unused_input='ignore', givens={x: validate_x, y: validate_y})
    test_model = theano.function([index], mlp.errors(y), on_unused_input='ignore', givens={x: test_x, y: test_y})

    print 'training with %s...' % update_rule
    return main_loop(n_epochs=n_epochs, train_model=train_model, val_model=val_model, test_model=test_model,
                     set_zero=set_zero, zero_vec=zero_vec, n_train_batches=n_train_batches, shuffle_batch=shuffle_batch,
                     validation_only=validation_only)

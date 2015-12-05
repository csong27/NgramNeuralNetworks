from neural_network import *
from io_utils.load_data import *
from doc_embedding import read_matrices_kaggle_pickle
from path import Path
from sklearn.cross_validation import StratifiedShuffleSplit
import cPickle as pkl


def train_ngram_conv_net(
        datasets,
        input_shape,
        ngrams=(2, 1),
        ngram_activation=tanh,
        n_epochs=25,
        multi_kernel=True,
        concat_out=False,
        n_kernels=(4, 3),
        use_bias=False,
        ngram_bias=False,
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
        no_test_y=False,
        save_ngram=False,
        validation_only=False
):
    rng = np.random.RandomState(23455)
    if no_test_y:
        train_x, train_y, validate_x, validate_y, test_x = datasets
        test_y = np.asarray([])
    else:
        train_x, train_y, validate_x, validate_y, test_x, test_y = datasets

    print 'size of train, validation, test set are %d, %d, %d' % (train_y.shape[0], validate_y.shape[0], test_x.shape[0])

    train_x, train_y = shared_dataset((train_x, train_y))
    validate_x, validate_y = shared_dataset((validate_x, validate_y))
    test_x, test_y = shared_dataset((test_x, test_y))

    n_train_batches = train_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    print 'building network, number of mini-batches are %d...' % n_train_batches

    index = T.lscalar()  # index to a [mini]batch

    x = T.tensor3('x')
    y = T.ivector('y')

    # weather or not to use multiple kernels in the n gram layer
    if multi_kernel:
        ngram_net = MultiKernelNgramNetwork(rng=rng, input=x, input_shape=input_shape, ngrams=ngrams, n_kernels=n_kernels,
                                            activation=ngram_activation, mean=mean_pool, concat_out=concat_out)
    else:
        ngram_net = NgramNetwork(rng=rng, input=x, input_shape=input_shape, ngrams=ngrams, use_bias=ngram_bias,
                                 activation=ngram_activation)

    dim = input_shape[1]
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

    # functions for making prediction
    if no_test_y:
        predict_output = mlp.layers[-1].y_pred if dropout else mlp.logRegressionLayer.y_pred
        predict_model = theano.function([index], predict_output, on_unused_input='ignore', givens={x: test_x, y: test_y})

    # functions for getting document vectors
    if save_ngram:
        save_train = theano.function([index], ngram_net.output, on_unused_input='ignore', givens={x: train_x, y: train_y})
        save_validate = theano.function([index], ngram_net.output, on_unused_input='ignore', givens={x: validate_x, y: validate_y})
        save_test = theano.function([index], ngram_net.output, on_unused_input='ignore', givens={x: test_x, y: test_y})

    print 'training with %s...' % update_rule
    epoch = 0
    best_val_accuracy = 0
    test_accuracy = 0
    best_prediction = None
    while epoch < n_epochs:
        epoch += 1
        cost_list = []
        indices = np.random.permutation(range(n_train_batches)) if shuffle_batch else xrange(n_train_batches)
        for minibatch_index in indices:
            cost_mini = train_model(minibatch_index)
            cost_list.append(cost_mini)
        val_accuracy = 1 - val_model(epoch)
        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            if not no_test_y:
                test_accuracy = 1 - test_model(epoch)
            else:
                best_prediction = predict_model(epoch)
            # saving best pretrained vectors
            if save_ngram:
                saved_train = save_train(epoch)
                saved_validate = save_validate(epoch)
                saved_test = save_test(epoch)
        cost_epoch = np.mean(cost_list)
        print 'epoch %i, train cost %f, validate accuracy %f' % (epoch, cost_epoch, val_accuracy * 100.)
    if validation_only:
        return best_val_accuracy
    if save_ngram:
        return saved_train, saved_validate, saved_test
    if not no_test_y:
        print "\nbest test accuracy is %f" % test_accuracy
        return test_accuracy
    else:
        return best_prediction


def train_ngram_net_embedding(
        datasets,
        U,
        input_shape,
        non_static=True,
        ngrams=(2, 1),
        ngram_out=(200, 100),
        ngram_activation=tanh,
        n_epochs=25,
        multi_kernel=True,
        concat_out=False,
        n_kernels=(4, 3),
        use_bias=False,
        ngram_bias=False,
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
        no_test_y=False,
        save_ngram=False,
        validation_only=False
):
    rng = np.random.RandomState(23455)

    train_x, train_y, validate_x, validate_y, test_x, test_y = datasets

    print 'size of train, validation, test set are %d, %d, %d' % (train_y.shape[0], validate_y.shape[0], test_x.shape[0])

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
    # weather or not to use multiple kernels in the n gram layer
    if multi_kernel:
        ngram_net = MultiKernelNgramNetwork(rng=rng,
                                            input=ngram_input,
                                            input_shape=input_shape,
                                            ngrams=ngrams,
                                            ngram_out=ngram_out,
                                            n_kernels=n_kernels,
                                            activation=ngram_activation,
                                            mean=mean_pool,
                                            concat_out=concat_out)
    else:
        ngram_net = NgramNetwork(rng=rng,
                                 input=ngram_input,
                                 input_shape=input_shape,
                                 ngrams=ngrams,
                                 use_bias=ngram_bias,
                                 activation=ngram_activation)

    dim = input_shape[1]
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

    params = ngram_net.params + mlp.params    # learning word vectors as well
    if non_static:
        params += [Words]

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

    # functions for making prediction
    if no_test_y:
        predict_output = mlp.layers[-1].y_pred if dropout else mlp.logRegressionLayer.y_pred
        predict_model = theano.function([index], predict_output, on_unused_input='ignore', givens={x: test_x, y: test_y})

    # functions for getting document vectors
    if save_ngram:
        save_train = theano.function([index], ngram_net.output, on_unused_input='ignore', givens={x: train_x, y: train_y})
        save_validate = theano.function([index], ngram_net.output, on_unused_input='ignore', givens={x: validate_x, y: validate_y})
        save_test = theano.function([index], ngram_net.output, on_unused_input='ignore', givens={x: test_x, y: test_y})

    print 'training with %s...' % update_rule
    epoch = 0
    best_val_accuracy = 0
    test_accuracy = 0
    best_prediction = None
    while epoch < n_epochs:
        epoch += 1
        cost_list = []
        # shuffle mini-batch if specified
        indices = np.random.permutation(range(n_train_batches)) if shuffle_batch else xrange(n_train_batches)
        for minibatch_index in indices:
            cost_mini = train_model(minibatch_index)
            cost_list.append(cost_mini)
            set_zero(zero_vec)  # reset the zero vectors
        val_accuracy = 1 - val_model(epoch)
        # get a best result
        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            if not no_test_y:
                test_accuracy = 1 - test_model(epoch)
            else:
                best_prediction = predict_model(epoch)
            # saving best pretrained vectors
            if save_ngram:
                saved_train = save_train(epoch)
                saved_validate = save_validate(epoch)
                saved_test = save_test(epoch)
        cost_epoch = np.mean(cost_list)
        print 'epoch %i, train cost %f, validate accuracy %f' % (epoch, cost_epoch, val_accuracy * 100.)

    # return values
    if validation_only:
        return best_val_accuracy
    if save_ngram:
        return saved_train, saved_validate, saved_test
    if not no_test_y:
        print "\nbest test accuracy is %f" % test_accuracy
        return test_accuracy
    else:
        return best_prediction

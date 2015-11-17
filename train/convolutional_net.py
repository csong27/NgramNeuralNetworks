from neural_network import *
from doc_embedding import *


def train_ngram_conv_net(
        datasets,
        ngrams=(2, 1),
        dim=50,
        ngram_activation=tanh,
        n_epochs=25,
        use_bias=False,
        ngram_bias=False,
        shuffle_batch=True,
        batch_size=50,
        dropout=True,
        n_hidden=100,
        n_out=2,
        activation=relu,
        dropout_rate=0.3,
        update_rule='adadelta',
        lr_rate=0.01,
        momentum_ratio=0.9
):
    rng = np.random.RandomState(23455)
    train_x, train_y, validate_x, validate_y, test_x, test_y = datasets

    print 'size of train, validation, test set are %d, %d, %d' %(train_y.shape[0], validate_y.shape[0], test_y.shape[0])

    train_x, train_y = shared_dataset((train_x, train_y))
    validate_x, validate_y = shared_dataset((validate_x, validate_y))
    test_x, test_y = shared_dataset((test_x, test_y))

    n_train_batches = train_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    print 'building network, number of mini-batches are %d...' % n_train_batches

    index = T.lscalar()  # index to a [mini]batch

    x = T.tensor3('x')
    y = T.ivector('y')

    ngram_net = NgramNetwork(rng=rng, input=x, dim=dim, ngrams=ngrams, activation=ngram_activation, use_bias=ngram_bias)

    mlp_input = ngram_net.output

    if dropout:
        layer_sizes = [dim, n_hidden, n_out]
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
            n_in=dim,
            n_hidden=n_hidden,
            n_out=n_out,
            activation=activation
        )
        cost = mlp.negative_log_likelihood(y)

    params = ngram_net.params + mlp.params

    if update_rule == 'adadelta':
        grad_updates = adadelta(loss_or_grads=cost, params=params, learning_rate=lr_rate, rho=0.95, epsilon=1e-6)
    elif update_rule == 'adagrad':
        grad_updates = adagrad(loss_or_grads=cost, params=params, learning_rate=lr_rate, epsilon=1e-6)
    elif update_rule == 'adam':
        grad_updates = adam(loss_or_grads=cost, params=params, learning_rate=lr_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif update_rule == 'momentum':
        grad_updates = momentum(loss_or_grads=cost, params=params, momentum=momentum_ratio, learning_rate=lr_rate)
    elif update_rule == 'sgd':
        grad_updates = sgd(loss_or_grads=cost, params=params, learning_rate=lr_rate)
    else:
        raise NotImplementedError("This optimization method is not implemented %s" % update_rule)

    train_model = theano.function([index], cost, updates=grad_updates,
                                  givens={
                                      x: train_x[index * batch_size:(index + 1) * batch_size],
                                      y: train_y[index * batch_size:(index + 1) * batch_size]
                                  })

    val_model = theano.function([index], mlp.errors(y), on_unused_input='ignore', givens={x: validate_x, y: validate_y})

    test_model = theano.function([index], mlp.errors(y), on_unused_input='ignore', givens={x: test_x, y: test_y})

    print 'training with %s...' % update_rule
    epoch = 0
    best_val_accuracy = 0
    test_accuracy = 0
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
            test_accuracy = 1 - test_model(epoch)
        cost_epoch = np.mean(cost_list)
        print 'epoch %i, train cost %f, validate accuracy %f' % (epoch, cost_epoch, val_accuracy * 100.)

    print "\nbest test accuracy is %f" % test_accuracy
    return test_accuracy


def wrapper(data=TREC):
    train_x, train_y, validate_x, validate_y, test_x, test_y = read_matrices_pickle(google=False, data=data, cv=False)
    dim = train_x[0].shape[1]
    n_out = len(np.unique(test_y))
    shuffle_indices = np.random.permutation(train_x.shape[0])
    datasets = (train_x[shuffle_indices], train_y[shuffle_indices], validate_x, validate_y, test_x, test_y)
    train_ngram_conv_net(
        datasets=datasets,
        ngrams=(2, 1),
        use_bias=True,
        ngram_bias=False,
        dim=dim,
        lr_rate=0.01,
        dropout=True,
        dropout_rate=0.5,
        n_hidden=200,
        n_out=n_out,
        activation=tanh,
        ngram_activation=leaky_relu,
        batch_size=50,
        update_rule='adagrad'
    )


def wrapper_yelp():
    dim = 50
    cutoff = 50
    datasets = get_document_matrices_yelp(dim=dim, cutoff=cutoff)
    train_ngram_conv_net(
        datasets=datasets,
        ngrams=(2, 1),
        use_bias=False,
        ngram_bias=False,
        dim=dim,
        lr_rate=0.001,
        dropout=True,
        dropout_rate=0.5,
        n_hidden=200,
        activation=relu,
        ngram_activation=leaky_relu,
        batch_size=50,
        update_rule='adagrad'
    )


if __name__ == '__main__':
    wrapper()
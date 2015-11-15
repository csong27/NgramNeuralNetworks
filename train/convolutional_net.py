from neural_network import *
from doc_embedding import get_document_matrices


def train_bigram_conv_net(
        n_epochs=25,
        shuffle_batch=True,
        batch_size=50,
        dim=50,
        dropout=False,
        n_hidden=50,
        activation=tanh,
        dropout_rate=0.1,
        lr_decay=0.95,
        epsilon=1e-6,
        sqr_norm_lim=9,
):
    rng = np.random.RandomState(23455)

    train_x, train_y, validate_x, validate_y, test_x, test_y = get_document_matrices(dim=dim, cutoff=50)
    train_x, train_y = shared_dataset((train_x, train_y))
    validate_x, validate_y = shared_dataset((validate_x, validate_y))
    test_x, test_y = shared_dataset((test_x, test_y))

    n_train_batches = train_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    print 'building network, number of mini-batches are %d' % n_train_batches

    index = T.lscalar()  # index to a [mini]batch

    x = T.tensor3('x')
    y = T.ivector('y')

    bigram_layer = BigramLayer(rng=rng, input=x, n_in=dim, n_out=dim)
    mlp_input = bigram_layer.output

    if dropout:
        mlp = MLPDropout(
            rng=rng,
            input=mlp_input,
            layer_sizes=[n_hidden, 5],
            dropout_rates=[dropout_rate],
            activations=[activation]
        )
        cost = mlp.dropout_negative_log_likelihood(y)
    else:
        mlp = MLP(
            rng=rng,
            input=mlp_input,
            n_in=dim,
            n_hidden=n_hidden,
            n_out=5,
            activation=activation
        )
        cost = mlp.negative_log_likelihood(y)

    params = bigram_layer.params + mlp.params

    grad_updates = updates_adadelta(params, cost, lr_decay, epsilon, sqr_norm_lim)

    train_model = theano.function([index], cost, updates=grad_updates,
                                  givens={
                                      x: train_x[index * batch_size:(index + 1) * batch_size],
                                      y: train_y[index * batch_size:(index + 1) * batch_size]
                                  })

    val_model = theano.function([index], mlp.errors(y), on_unused_input='ignore', givens={x: validate_x, y: validate_y})

    test_model = theano.function([index], mlp.errors(y), on_unused_input='ignore', givens={x: test_x, y: test_y})

    print '... training'
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

        cost_epoch = np.mean(cost_list)
        val_error = val_model(epoch)
        val_accuracy = 1 - np.mean(val_error)
        print 'epoch %i, train cost %f, validate accuracy %f' % (epoch, cost_epoch, val_accuracy * 100.)
        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            test_loss = test_model(epoch)
            test_accuracy = 1 - test_loss

    print test_accuracy

if __name__ == '__main__':
    train_bigram_conv_net()

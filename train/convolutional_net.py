from neural_network import *
from utils.load_data import *
from doc_embedding import read_matrices_kaggle_pickle
from path import Path
import cPickle as pkl


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
        momentum_ratio=0.9,
        no_test_y=False,
        save_ngram=False,
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

    # functions for training
    train_model = theano.function([index], cost, updates=grad_updates,
                                  givens={
                                      x: train_x[index * batch_size:(index + 1) * batch_size],
                                      y: train_y[index * batch_size:(index + 1) * batch_size]
                                  })
    val_model = theano.function([index], mlp.errors(y), on_unused_input='ignore', givens={x: validate_x, y: validate_y})
    test_model = theano.function([index], mlp.errors(y), on_unused_input='ignore', givens={x: test_x, y: test_y})

    # functions for making prediction
    predict_output = mlp.layers[-1].y_pred if dropout else mlp.logRegressionLayer.y_pred
    predict_model = theano.function([index], predict_output, on_unused_input='ignore', givens={x: test_x, y: test_y})

    # functions for getting document vectors
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
    if save_ngram:
        return saved_train, saved_validate, saved_test
    if not no_test_y:
        print "\nbest test accuracy is %f" % test_accuracy
        return test_accuracy
    else:
        return best_prediction


def save_ngram_vectors(data=SST_KAGGLE):
    if data == SST_KAGGLE:
        train_x, train_y, validate_x, validate_y, test_x = read_matrices_kaggle_pickle()
        datasets = (train_x, train_y, validate_x, validate_y, test_x)
        no_test_y = True
    else:
        raise NotImplementedError

    dim = train_x[0].shape[1]
    n_out = len(np.unique(validate_y))
    saved_train, saved_validate, saved_test = train_ngram_conv_net(
        datasets=datasets,
        ngrams=(2,),
        use_bias=True,
        n_epochs=20,
        ngram_bias=False,
        dim=dim,
        lr_rate=0.1,
        n_out=n_out,
        dropout=True,
        dropout_rate=0.5,
        n_hidden=100,
        activation=leaky_relu,
        ngram_activation=leaky_relu,
        batch_size=200,
        update_rule='adagrad',
        no_test_y=no_test_y,
        save_ngram=True
    )

    print saved_train.shape

    save_path = "D:/data/nlpdata/pickled_data/doc2vec/"
    save_path += data + "_ngram.pkl"
    print "saving doc2vec to %s" % save_path

    f = open(Path(save_path), "wb")
    pkl.dump((saved_train, saved_validate, saved_test), f, -1)
    f.close()


def read_ngram_vectors(data=SST_KAGGLE):
    save_path = "D:/data/nlpdata/pickled_data/doc2vec/"
    save_path += data + "_ngram.pkl"
    print "reading doc2vec from %s" % save_path

    f = open(Path(save_path), "rb")
    saved_train, saved_validate, saved_test = pkl.load(f)
    f.close()

    return saved_train, saved_validate, saved_test

if __name__ == '__main__':
    save_ngram_vectors()

import cPickle as pkl
from path import Path
from sklearn.cross_validation import StratifiedShuffleSplit
from neural_network import *
from utils import save_csv
from utils.load_data import *
from utils.pickled_feature import get_concatenated_document_vectors


def train_dropout_net(
        datasets,
        dim=50,
        n_epochs=25,
        use_bias=False,
        shuffle_batch=True,
        batch_size=50,
        dropout=True,
        n_hidden=[100],
        n_out=2,
        activations=[relu],
        dropout_rates=[0.3],
        update_rule='adadelta',
        lr_rate=0.01,
        momentum_ratio=0.9,
        no_test_y=False,
        save_prob=False
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

    x = T.matrix('x')
    y = T.ivector('y')
    mlp_input = x

    if dropout:
        layer_sizes = [dim] + n_hidden + [n_out]
        mlp = MLPDropout(
            rng=rng,
            input=mlp_input,
            layer_sizes=layer_sizes,
            dropout_rates=dropout_rates,
            activations=activations,
            use_bias=use_bias
        )
        cost = mlp.dropout_negative_log_likelihood(y)
    else:
        mlp = MLP(
            rng=rng,
            input=mlp_input,
            n_in=dim,
            n_hidden=n_hidden[0],
            n_out=n_out,
            activation=activations[0]
        )
        cost = mlp.negative_log_likelihood(y)

    params = mlp.params

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

    predict_output = mlp.layers[-1].y_pred if dropout else mlp.logRegressionLayer.y_pred
    predict_model = theano.function([index], predict_output, on_unused_input='ignore', givens={x: test_x, y: test_y})

    if save_prob:
        prob = mlp.layers[-1].p_y_given_x if dropout else mlp.logRegressionLayer.p_y_given_x
        train_prob_fn = theano.function([index], prob, on_unused_input='ignore', givens={x: train_x})
        validate_prob_fn = theano.function([index], prob, on_unused_input='ignore', givens={x: validate_x})
        test_prob_fn = theano.function([index], prob, on_unused_input='ignore', givens={x: test_x})

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
            if save_prob:   # saving the predicted score for each class
                train_prob = train_prob_fn(epoch)
                validate_prob = validate_prob_fn(epoch)
                test_prob = test_prob_fn(epoch)

        cost_epoch = np.mean(cost_list)
        print 'epoch %i, train cost %f, validate accuracy %f' % (epoch, cost_epoch, val_accuracy * 100.)
    if not no_test_y:
        print "\nbest test accuracy is %f" % test_accuracy
        return test_accuracy
    elif save_prob:
        return train_prob, validate_prob, test_prob
    else:
        return best_prediction


def wrapper_kaggle(epochs=40, validate_ratio=0.1, save_prob=True):
    old_train_x, test_x = get_concatenated_document_vectors(data=SST_KAGGLE)
    _, old_train_y, _ = read_sst_kaggle_pickle()
    old_train_y = np.asarray(old_train_y)
    # split train validate data
    sss_indices = StratifiedShuffleSplit(y=old_train_y, n_iter=1, test_size=validate_ratio, random_state=42)
    for indices in sss_indices:
        train_index, test_index = indices
    train_x = old_train_x[train_index]
    validate_x = old_train_x[test_index]
    train_y = old_train_y[train_index]
    validate_y = old_train_y[test_index]
    # get dataset info
    dim = train_x[0].shape[0]
    n_out = len(np.unique(validate_y))
    datasets = (train_x, train_y, validate_x, validate_y, test_x)

    n_layers = 1

    print "input dimension is %d, output dimension is %d" % (dim, n_out)

    return_val = train_dropout_net(
        datasets=datasets,
        use_bias=True,
        n_epochs=epochs,
        dim=dim,
        lr_rate=0.02,
        n_out=n_out,
        dropout=True,
        dropout_rates=[0.5] * n_layers,
        n_hidden=[300] * n_layers,
        activations=[leaky_relu] * n_layers,
        batch_size=100,
        update_rule='adagrad',
        no_test_y=True,
        save_prob=save_prob
    )
    if not save_prob:
        save_csv(return_val)
    else:
        train_prob, validate_prob, test_prob = return_val
        saved_train_prob = np.zeros((old_train_x.shape[0], n_out))
        saved_train_prob[train_index] = train_prob
        saved_train_prob[test_index] = validate_prob
        save_path = "D:/data/nlpdata/pickled_data/" + SST_KAGGLE + "_prob.pkl"
        print "saving probability feature to %s" % save_path

        f = open(Path(save_path), "wb")
        pkl.dump((saved_train_prob, test_prob), f, -1)
        f.close()


if __name__ == '__main__':
    wrapper_kaggle()
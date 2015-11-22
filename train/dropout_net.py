from neural_network import *
from doc_embedding import *
from utils.load_data import *
from convolutional_net import read_ngram_vectors
from sklearn.cross_validation import train_test_split
from path import Path
import numpy as np


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
        cost_epoch = np.mean(cost_list)
        print 'epoch %i, train cost %f, validate accuracy %f' % (epoch, cost_epoch, val_accuracy * 100.)
    if not no_test_y:
        print "\nbest test accuracy is %f" % test_accuracy
        return test_accuracy
    else:
        return best_prediction


def get_concatenated_document_vectors(data=SST_KAGGLE):
    train_x_1, test_x_1 = read_doc2vec_pickle(dm=True, concat=False, data=data)
    train_x_2, test_x_2 = read_doc2vec_pickle(dm=False, concat=False, data=data)
    train_x_3, test_x_3 = read_ngram_vectors(data=data)

    train_x = np.concatenate((train_x_1, train_x_2), axis=1)
    test_x = np.concatenate((test_x_1, test_x_2), axis=1)

    return train_x, test_x


def wrapper_kaggle(validate_ratio=0.2):
    train_x, test_x = get_concatenated_document_vectors(data=SST_KAGGLE)
    _, train_y, _ = read_sst_kaggle_pickle()
    train_x, validate_x, train_y, validate_y = train_test_split(train_x, train_y, test_size=validate_ratio,
                                                                random_state=42, stratify=train_y)

    train_y = np.asarray(train_y)
    validate_y = np.asarray(validate_y)

    dim = train_x[0].shape[0]
    n_out = len(np.unique(validate_y))
    datasets = (train_x, train_y, validate_x, validate_y, test_x)

    n_layers = 1

    print "input dimension is %d, output dimension is %d" % (dim, n_out)

    best_prediction = train_dropout_net(
        datasets=datasets,
        use_bias=True,
        n_epochs=40,
        dim=dim,
        lr_rate=0.5,
        n_out=n_out,
        dropout=True,
        dropout_rates=[0.5] * n_layers,
        n_hidden=[300] * n_layers,
        activations=[leaky_relu] * n_layers,
        batch_size=100,
        update_rule='adagrad',
        no_test_y=True
    )

    import csv
    save_path = Path('C:/Users/Song/Course/571/hw3/kaggle_result.csv')
    with open(save_path, 'wb') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['PhraseId', 'Sentiment'])
        phrase_ids = np.arange(156061, 222353)
        for phrase_id, sentiment in zip(phrase_ids, best_prediction):
            writer.writerow([phrase_id, sentiment])

if __name__ == '__main__':
    wrapper_kaggle()
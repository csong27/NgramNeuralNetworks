from ngram_net import train_ngram_net
from ngram_reccurent_net import train_ngram_rec_net
from neural_network.non_linear import *
from io_utils.load_data import *
import numpy as np
from train import prepare_datasets


def wrapper_ngram(data=TREC, resplit=True, validate_ratio=0.2):
    train_x, train_y, validate_x, validate_y, test_x, test_y, \
    W, mask = prepare_datasets(data, resplit=resplit, validation_ratio=validate_ratio)
    # get input shape
    input_shape = (train_x[0].shape[0], W.shape[1])
    print "input data shape", input_shape
    n_out = len(np.unique(test_y))
    shuffle_indices = np.random.permutation(train_x.shape[0])
    datasets = (train_x[shuffle_indices], train_y[shuffle_indices], validate_x, validate_y, test_x, test_y)
    test_accuracy = train_ngram_net(
        U=W,
        datasets=datasets,
        n_epochs=10,
        ngrams=(3, 2),
        ngram_out=(150, 50),
        non_static=False,
        input_shape=input_shape,
        concat_out=True,
        n_kernels=(8, 16),
        use_bias=False,
        lr_rate=0.02,
        dropout=True,
        dropout_rate=0.5,
        n_hidden=600,
        n_out=n_out,
        ngram_activation=leaky_relu,
        activation=leaky_relu,
        batch_size=50,
        l2_ratio=1e-5,
        update_rule='adagrad',
        skip_gram=False,
    )
    return test_accuracy


def wrapper_rec(data=SST_SENT_POL, resplit=True, validate_ratio=0.2, rec_type='lstm'):
    train_x, train_y, validate_x, validate_y, test_x, test_y, \
    W, mask = prepare_datasets(data, resplit=resplit, validation_ratio=validate_ratio)
    # get input shape
    input_shape = (train_x[0].shape[0], W.shape[1])
    print "input data shape", input_shape
    n_out = len(np.unique(test_y))
    shuffle_indices = np.random.permutation(train_x.shape[0])
    datasets = (train_x[shuffle_indices], train_y[shuffle_indices], validate_x, validate_y, test_x, test_y)
    test_accuracy = train_ngram_rec_net(
        U=W,
        non_static=True,
        datasets=datasets,
        n_epochs=20,
        ngrams=(2, 2),
        concat_out=False,
        input_shape=input_shape,
        n_kernels=(4, 4),
        ngram_out=(300, 250),
        lr_rate=0.015,
        dropout_rate=0.5,
        rec_hidden=250,
        mlp_hidden=200,
        mlp=True,
        n_out=n_out,
        ngram_activation=tanh,
        rec_activation=tanh,
        mlp_activation=tanh,
        batch_size=50,
        update_rule='adagrad',
        rec_type=rec_type,
        l2_ratio=1e-4,
        mask=mask,
        clipping=1,
        skip_gram=True,
        word_dropout_rate=0.3
    )
    return test_accuracy


def wrapper_reversed_rec(data=SST_SENT_POL, resplit=True, validate_ratio=0.2, rec_type='lstm'):
    train_x, train_y, validate_x, validate_y, test_x, test_y, \
    W, mask = prepare_datasets(data, resplit=resplit, validation_ratio=validate_ratio)
    # get input shape
    input_shape = (train_x[0].shape[0], W.shape[1])
    print "input data shape", input_shape
    n_out = len(np.unique(test_y))
    shuffle_indices = np.random.permutation(train_x.shape[0])
    datasets = (train_x[shuffle_indices], train_y[shuffle_indices], validate_x, validate_y, test_x, test_y)
    test_accuracy = train_ngram_rec_net(
        reverse=True,
        U=W,
        non_static=False,
        datasets=datasets,
        n_epochs=30,
        ngrams=(1, 2),
        input_shape=input_shape,
        n_kernels=(4, 4),
        ngram_out=(300, 250),
        lr_rate=0.025,
        dropout_rate=0.,
        concat_out=False,
        rec_hidden=150,
        mlp_hidden=200,
        n_out=n_out,
        ngram_activation=leaky_relu,
        mlp_activation=leaky_relu,
        rec_activation=tanh,
        batch_size=20,
        update_rule='adadelta',
        rec_type=rec_type,
        clipping=1,
        l2_ratio=1e-5,
        mask=mask,
        mlp=True,
        skip_gram=False,
        bidirection=True
    )
    return test_accuracy


def error_analysis(data=SST_SENT_POL):
    train_x, train_y, validate_x, validate_y, test_x, test_y, \
    W, mask = prepare_datasets(data, resplit=False, validation_ratio=0.0)
    # get input shape
    input_shape = (train_x[0].shape[0], W.shape[1])
    print "input data shape", input_shape
    n_out = len(np.unique(test_y))
    shuffle_indices = np.random.permutation(train_x.shape[0])
    datasets = (train_x[shuffle_indices], train_y[shuffle_indices], validate_x, validate_y, test_x, test_y)
    best_prediction = train_ngram_net(
        U=W,
        datasets=datasets,
        n_epochs=10,
        ngrams=(1, 2),
        ngram_out=(300, 250),
        non_static=False,
        input_shape=input_shape,
        concat_out=False,
        n_kernels=(4, 4),
        use_bias=False,
        lr_rate=0.02,
        dropout=True,
        dropout_rate=0.2,
        n_hidden=250,
        n_out=n_out,
        ngram_activation=leaky_relu,
        activation=leaky_relu,
        batch_size=50,
        l2_ratio=1e-5,
        update_rule='adagrad',
        skip_gram=False,
        predict=True
    )
    raw_datasets = load_raw_datasets(datasets=data)
    _, _, validate_raw, _, _, _ = raw_datasets
    from collections import Counter
    errors = []
    for i in xrange(len(best_prediction)):
        if best_prediction[i] != validate_y[i]:
            errors.append("%d & %d" % (validate_y[i], best_prediction[i]))
            print validate_y[i], best_prediction[i], " ".join(validate_raw[i])
    errors = Counter(errors)
    print errors.most_common(10)

if __name__ == '__main__':
    # for data in [SST_SENT_POL, SST_SENT, TREC]:
    #     for rec in ['lstm', 'gru']:
    wrapper_reversed_rec(data=SST_SENT_POL)

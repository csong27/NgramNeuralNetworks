from ngram_net import train_ngram_net
from ngram_reccurent_net import train_ngram_rec_net
from io_utils.word_index import read_word2index_data
from neural_network.non_linear import *
from io_utils.load_data import *
from sklearn.cross_validation import StratifiedShuffleSplit
import numpy as np


def prepare_datasets(data, resplit=True, validation_ratio=0.2):
    datasets, W, mask = read_word2index_data(data=data, google=True, cv=False)
    train_x, train_y, validate_x, validate_y, test_x, test_y = datasets
    if data == TREC and resplit:
        train_x, train_y, validate_x, validate_y, mask = resplit_train_data(train_x, train_y, validate_x, validate_y,
                                                                            validation_ratio, mask=mask)
    return train_x, train_y, validate_x, validate_y, test_x, test_y, W, mask


def resplit_train_data(train_x, train_y, validate_x, validate_y, validate_ratio, mask=None):
    all_x = np.concatenate((train_x, validate_x), axis=0)
    all_y = np.concatenate((train_y, validate_y))
    sss_indices = StratifiedShuffleSplit(y=all_y, n_iter=1, test_size=validate_ratio)
    for indices in sss_indices:
        train_index, test_index = indices
    train_x = all_x[train_index]
    validate_x = all_x[test_index]
    train_y = all_y[train_index]
    validate_y = all_y[test_index]
    if mask is not None:
        train_mask, validate_mask, test_mask = mask
        all_mask = np.concatenate((train_mask, validate_mask), axis=0)
        train_mask = all_mask[train_index]
        validate_mask = all_mask[test_index]
        mask = (train_mask, validate_mask, test_mask)
    return train_x, train_y, validate_x, validate_y, mask


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
        n_epochs=15,
        ngrams=(2, 1),
        ngram_out=(200, 100),
        non_static=True,
        input_shape=input_shape,
        concat_out=True,
        n_kernels=(4, 4),
        use_bias=False,
        lr_rate=0.02,
        dropout=True,
        dropout_rate=0.5,
        n_hidden=300,
        n_out=n_out,
        ngram_activation=leaky_relu,
        activation=leaky_relu,
        batch_size=50,
        l2_ratio=1e-5,
        update_rule='adagrad',
        skip_gram=True
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
        skip_gram=True
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
        n_epochs=20,
        ngrams=(3, 2),
        input_shape=input_shape,
        n_kernels=(4, 4),
        ngram_out=(250, 200),
        lr_rate=0.015,
        dropout_rate=0.5,
        concat_out=False,
        rec_hidden=300,
        mlp_hidden=300,
        n_out=n_out,
        ngram_activation=leaky_relu,
        mlp_activation=leaky_relu,
        rec_activation=leaky_relu,
        batch_size=50,
        update_rule='adagrad',
        rec_type=rec_type,
        clipping=1,
        l2_ratio=1e-5,
        mask=mask,
        mlp=True,
        skip_gram=True
    )
    return test_accuracy


if __name__ == '__main__':
    # for data in [SST_SENT_POL, SST_SENT, TREC]:
    #     for rec in ['lstm', 'gru']:
    wrapper_reversed_rec(data=SST_SENT_POL, rec_type='gru')

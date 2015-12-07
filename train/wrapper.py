from ngram_net import train_ngram_conv_net, train_ngram_net_embedding
from ngram_reccurent_net import train_ngram_rec_net
from io_utils.word_index import read_word2index_data
from neural_network.non_linear import *
from doc_embedding import *
from io_utils.load_data import *
import numpy as np


def resplit_train_data(train_x, train_y, validate_x, validate_y, validate_ratio):
    all_x = np.concatenate((train_x, validate_x), axis=0)
    all_y = np.concatenate((train_y, validate_y))
    from sklearn.cross_validation import train_test_split
    train_x, validate_x, train_y, validate_y = train_test_split(all_x, all_y, test_size=validate_ratio, stratify=all_y)
    return train_x, train_y, validate_x, validate_y


def wrapper(data=SST_SENT_POL):
    train_x, train_y, validate_x, validate_y, test_x, test_y = read_matrices_pickle(google=True, data=data, cv=False,
                                                                                    huge=False)
    # get input shape
    input_shape = train_x[0].shape
    print "input data shape", input_shape
    n_out = len(np.unique(test_y))
    shuffle_indices = np.random.permutation(train_x.shape[0])
    datasets = (train_x[shuffle_indices], train_y[shuffle_indices], validate_x, validate_y, test_x, test_y)
    test_accuracy = train_ngram_conv_net(
        datasets=datasets,
        n_epochs=15,
        ngrams=(1, 2),
        input_shape=input_shape,
        ngram_bias=False,
        multi_kernel=True,
        concat_out=False,
        n_kernels=(4, 4),
        use_bias=True,
        lr_rate=0.02,
        dropout=True,
        dropout_rate=0.5,
        n_hidden=400,
        n_out=n_out,
        ngram_activation=leaky_relu,
        activation=leaky_relu,
        batch_size=50,
        update_rule='adagrad'
    )
    return test_accuracy


def wrapper_word2index(data=TREC, resplit=True, validate_ratio=0.2):
    datasets, W, _ = read_word2index_data(data=data, google=True, cv=False)
    train_x, train_y, validate_x, validate_y, test_x, test_y = datasets
    if data == TREC and resplit:
        train_x, train_y, validate_x, validate_y = resplit_train_data(train_x, train_y, validate_x, validate_y, validate_ratio)
    # get input shape
    input_shape = (train_x[0].shape[0], W.shape[1])
    print "input data shape", input_shape
    n_out = len(np.unique(test_y))
    shuffle_indices = np.random.permutation(train_x.shape[0])
    datasets = (train_x[shuffle_indices], train_y[shuffle_indices], validate_x, validate_y, test_x, test_y)
    test_accuracy = train_ngram_net_embedding(
        U=W,
        datasets=datasets,
        n_epochs=15,
        ngrams=(1, 2),
        ngram_out=(300, 300),
        non_static=False,
        input_shape=input_shape,
        ngram_bias=False,
        multi_kernel=True,
        concat_out=False,
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
        update_rule='adagrad'
    )
    return test_accuracy


def wrapper_rec(data=SST_SENT_POL, rec_type='lstm'):
    datasets, W, mask = read_word2index_data(data=data, google=True, cv=False)
    train_x, train_y, validate_x, validate_y, test_x, test_y = datasets
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
        n_epochs=30,
        ngrams=(1, 2),
        input_shape=input_shape,
        n_kernels=(4, 4),
        ngram_out=(300, 250),
        lr_rate=0.02,
        dropout_rate=0.,
        n_hidden=200,
        n_out=n_out,
        ngram_activation=leaky_relu,
        batch_size=20,
        update_rule='adagrad',
        rec_type=rec_type,
        pool=True,
        mask=mask
    )
    return test_accuracy


if __name__ == '__main__':
    wrapper_rec(rec_type='lstm')
    print '\ndone\n'
    wrapper_rec(rec_type='gru')

from ngram_net import train_ngram_conv_net, train_ngram_net_embedding
from io_utils.word_index import read_word2index_data
from neural_network.non_linear import *
from doc_embedding import *
from io_utils.load_data import *
import numpy as np


def wrapper(data=SST_SENT_POL):
    train_x, train_y, validate_x, validate_y, test_x, test_y = read_matrices_pickle(google=False, data=data, cv=False,
                                                                                    huge=True)
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


def wrapper_word2index(data=SST_SENT_POL):
    datasets, W = read_word2index_data(data=data, google=True, cv=False)
    train_x, train_y, validate_x, validate_y, test_x, test_y = datasets
    # get input shape
    input_shape = (train_x[0].shape[0], W.shape[1])
    print "input data shape", input_shape
    n_out = len(np.unique(test_y))
    shuffle_indices = np.random.permutation(train_x.shape[0])
    datasets = (train_x[shuffle_indices], train_y[shuffle_indices], validate_x, validate_y, test_x, test_y)
    test_accuracy = train_ngram_net_embedding(
        U=W,
        datasets=datasets,
        n_epochs=20,
        ngrams=(2, 1),
        non_static=False,
        input_shape=input_shape,
        ngram_bias=False,
        multi_kernel=True,
        concat_out=False,
        n_kernels=(8, 8),
        use_bias=False,
        lr_rate=0.025,
        dropout=True,
        dropout_rate=0.3,
        n_hidden=300,
        n_out=n_out,
        ngram_activation=leaky_relu,
        activation=leaky_relu,
        batch_size=50,
        update_rule='adagrad'
    )
    return test_accuracy


if __name__ == '__main__':
    wrapper_word2index()

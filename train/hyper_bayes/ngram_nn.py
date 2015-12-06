from neural_network.non_linear import *
from train.ngram_net import train_ngram_net_embedding
from io_utils.load_data import *
from io_utils.word_index import read_word2index_data
import numpy as np


def ngram_wrapper(
        data=SST_SENT_POL,
        n_epochs=25,
        ngrams=(1, 2),
        multi_kernel=True,
        n_kernels=(4, 3),
        ngram_out=(300, 200),
        use_bias=False,
        batch_size=50,
        dropout=True,
        n_hidden=100,
        ngram_layers=2,
        dropout_rate=0.3,
        lr_rate=0.01,
        mean_pool=False,
        non_static=True,
        l2_ratio=1
):
    # getting the datasets
    datasets, W, _ = read_word2index_data(data=data, google=True, cv=False)
    train_x, train_y, validate_x, validate_y, test_x, test_y = datasets
    # get input shape
    input_shape = (train_x[0].shape[0], W.shape[1])
    print "input data shape", input_shape
    n_out = len(np.unique(test_y))
    shuffle_indices = np.random.permutation(train_x.shape[0])
    datasets = (train_x[shuffle_indices], train_y[shuffle_indices], validate_x, validate_y, test_x, test_y)
    # network configuration
    l2_ratio /= 1e4
    lr_rate /= 1e2
    n_epochs *= 2
    batch_size *= 10
    n_hidden *= 10
    dropout_rate /= 10.0
    # ngram layers configurations
    n_kernels = tuple(n_kernels[:ngram_layers])
    ngrams = tuple(ngrams[:ngram_layers])
    ngram_out = tuple(ngram_out[:ngram_layers])
    validation_accuracy = train_ngram_net_embedding(
        U=W,
        datasets=datasets,
        n_epochs=n_epochs,
        ngrams=ngrams,
        non_static=non_static,
        input_shape=input_shape,
        ngram_out=ngram_out,
        ngram_bias=False,
        multi_kernel=multi_kernel,
        concat_out=False,
        n_kernels=n_kernels,
        use_bias=use_bias,
        lr_rate=lr_rate,
        mean_pool=mean_pool,
        dropout=dropout,
        dropout_rate=dropout_rate,
        n_hidden=n_hidden,
        n_out=n_out,
        l2_ratio=l2_ratio,
        ngram_activation=leaky_relu,
        activation=leaky_relu,
        batch_size=batch_size,
        update_rule='adagrad',
        validation_only=True    # return the validation error to minimize
    )
    return 1 - validation_accuracy


# Write a function like this called 'main'
def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params
    return ngram_wrapper(
        lr_rate=params['lr_rate'][0],
        n_epochs=params['n_epochs'][0],
        batch_size=params['batch_size'][0],
        ngram_layers=params['ngram_layers'][0],
        use_bias=params['use_bias'][0],
        ngram_out=params['ngram_out'],  # passed as a list of three numbers
        n_kernels=params['n_kernels'],  # passed as a list of three numbers
        ngrams=params['ngrams'],  # passed as a list of three numbers
        dropout=params['dropout'][0],
        dropout_rate=params['dropout_rate'][0],
        n_hidden=params['n_hidden'][0],
        mean_pool=params['mean_pool'][0],
        non_static=params['non_static'][0],
        l2_ratio=params['l2_ratio']
    )

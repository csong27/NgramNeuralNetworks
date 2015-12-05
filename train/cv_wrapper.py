from doc_embedding import *
from sklearn.cross_validation import StratifiedKFold, train_test_split
from ngram_net import train_ngram_conv_net
from io_utils.load_data import *
from neural_network.non_linear import *
import numpy as np


def cross_validation(validation_ratio=0.1, data=ROTTEN_TOMATOES, shuffle=True):
    x, y = read_matrices_pickle(google=True, data=data, dim=300)
    # get input shape
    input_shape = x[0].shape
    n_out = len(np.unique(y))
    skf = StratifiedKFold(y, n_folds=10)
    accuracy_list = []
    for i, indices in enumerate(skf):
        print "\nat cross validation iter %i" % i
        print "\n**********************\n"
        train, test = indices
        train_x = x[train]
        train_y = y[train]
        test_x = x[test]
        test_y = y[test]
        train_x, validate_x, train_y, validate_y = train_test_split(train_x, train_y, test_size=validation_ratio,
                                                                    random_state=42, stratify=train_y)
        shuffle_indices = np.random.permutation(train_x.shape[0]) if shuffle else np.arange(train_x.shape[0])
        datasets = (train_x[shuffle_indices], train_y[shuffle_indices], validate_x, validate_y, test_x, test_y)
        test_accuracy = train_ngram_conv_net(
            datasets=datasets,
            n_epochs=15,
            ngrams=(2, 1),
            input_shape=input_shape,
            ngram_bias=False,
            multi_kernel=True,
            concat_out=False,
            n_kernels=(4, 4),
            use_bias=False,
            lr_rate=0.0175,
            dropout=True,
            dropout_rate=0.5,
            n_hidden=400,
            n_out=n_out,
            ngram_activation=leaky_relu,
            activation=leaky_relu,
            batch_size=50,
            update_rule='adagrad',
            mean_pool=False
        )
        accuracy_list.append(test_accuracy)

    print "\n**********************\nfinal result: %f" % np.mean(accuracy_list)

if __name__ == '__main__':
    cross_validation()

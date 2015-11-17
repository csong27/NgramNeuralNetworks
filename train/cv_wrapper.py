from doc_embedding import read_matrices_pickle
from sklearn.cross_validation import StratifiedKFold, train_test_split
from convolutional_net import train_ngram_conv_net
from neural_network.non_linear import *
import numpy as np


def cross_validation(validation_ratio=0.1, dim=200):
    x, y = read_matrices_pickle(google=False)
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
                                                                    random_state=42)
        datasets = (train_x, train_y, validate_x, validate_y, test_x, test_y)
        test_accuracy = train_ngram_conv_net(
            datasets=datasets,
            n_epochs=25,
            bigram=True,
            dim=dim,
            use_bias=True,
            lr_rate=0.01,
            dropout=True,
            dropout_rate=0.3,
            n_hidden=128,
            ngram_activation=tanh,
            activation=leaky_relu,
            batch_size=50,
            update_rule='adagrad'
        )
        accuracy_list.append(test_accuracy)

    print "\n**********************\nfinal result: %f" % np.mean(accuracy_list)

if __name__ == '__main__':
    cross_validation(dim=300)

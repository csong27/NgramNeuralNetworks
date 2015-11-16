from doc_embedding import get_document_matrices_rotten
from sklearn.cross_validation import StratifiedKFold, train_test_split
from convolutional_net import train_ngram_conv_net
from neural_network.non_linear import *
import numpy as np


def cross_validation(validation_ratio=0.1):
    x, y = get_document_matrices_rotten(dim=50, cutoff=50)
    skf = StratifiedKFold(y, n_folds=10)

    accuracy_list = []
    for i, indices in enumerate(skf):
        print "at cross validation iter %i" % i
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
            bigram=True,
            use_bias=False,
            lr_rate=0.001,
            dropout=False,
            dropout_rate=0.5,
            n_hidden=30,
            activation=relu,
            batch_size=1000,
            update_rule='adagrad'
        )
        accuracy_list.append(test_accuracy)

    print "\n**********************\nfinal result: %f" % np.mean(accuracy_list)

if __name__ == '__main__':
    cross_validation()
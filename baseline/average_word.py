from io_utils.load_data import *
from io_utils.save_kaggle_result import save_csv
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from doc_embedding import read_aggregated_vectors
import numpy as np


def train(data=SST_KAGGLE, alg='logcv'):
    train_x, train_y, test_x = read_aggregated_vectors(google=True, data=data)

    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    test_x = np.asarray(test_x)

    print "shape for training data is", train_x.shape

    if alg == 'svm':
        clf = SVC(verbose=1)
    elif alg == 'log':
        clf = LogisticRegression(verbose=1)
    elif alg == 'logcv':
        clf = LogisticRegressionCV(cv=5, verbose=1)
    else:
        raise NotImplementedError

    print "training..."
    clf.fit(train_x, train_y)
    # clf.fit(validate_x, validate_y)
    predicted = clf.predict(test_x)
    save_csv(predicted)

if __name__ == '__main__':
    train()
from sknn.mlp import Classifier, Layer
from machine_learning import get_data
from sklearn import metrics
import numpy as np
import sys
import logging

logging.basicConfig(
            format="%(message)s",
            level=logging.DEBUG,
            stream=sys.stdout)


def classification_neural_network(epochs=5):
    train_x, train_y, validate_x, validate_y, test_x, test_y = get_data(dim=300)

    train_x = np.asarray(train_x).astype('float64')
    train_y = np.asarray(train_y).astype('float64')

    validate_x = np.asarray(validate_x).astype('float64')
    validate_y = np.asarray(validate_y).astype('float64')

    test_x = np.asarray(test_x).astype('float64')
    test_y = np.asarray(test_y).astype('float64')

    nn = Classifier(
        layers=[Layer("Rectifier", units=100), Layer("Softmax")],
        valid_set=(validate_x, validate_y),
        batch_size=100,
        learning_rule="adagrad",
        regularize='L2',
        n_iter=epochs
    )

    nn.fit(train_x, train_y)

    print "\nfinal testing..."
    predicted = nn.predict(test_x)
    print nn.score(test_x, test_y)
    print(metrics.classification_report(test_y, predicted))

if __name__ == '__main__':
    classification_neural_network()

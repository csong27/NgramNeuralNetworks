from documentEmbedding.aggregateWordVector import get_aggregated_vectors
from sklearn import linear_model
from sklearn import metrics
import numpy as np


def get_regression_data(validate=False):
    train_x, train_y, validate_x, validate_y, test_x, test_y = get_aggregated_vectors()
    print "transforming data for raw regression..."
    if validate:
        return train_x, train_y, validate_x, validate_y, test_x, test_y
    train_x.extend(validate_x)
    train_y.extend(validate_y)
    return train_x, train_y, test_x, test_y


def raw_decode(prediction):
    for i in xrange(len(prediction)):
        raw_score = prediction[i]
        if raw_score < 1:
            prediction[i] = '1'
        elif raw_score > 5:
            prediction[i] = '5'
        else:
            prediction[i] = str(int(round(raw_score)))
    return prediction


def regression_baseline():
    train_x, train_y, test_x, test_y = get_regression_data(validate=False)
    train_x = np.asarray(train_x).astype('float64')
    train_y = np.asarray(train_y).astype('float64')
    test_x = np.asarray(test_x).astype('float64')
    test_y = np.asarray(test_y).astype('float64')

    regressor = linear_model.SGDRegressor(n_iter=20)

    print "training regression model..."
    regressor.fit(train_x, train_y)
    predicted = raw_decode(regressor.predict(test_x))

    print np.mean(predicted == test_y)
    print(metrics.classification_report(test_y, predicted))


if __name__ == '__main__':
    regression_baseline()
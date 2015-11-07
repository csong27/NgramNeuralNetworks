from machine_learning import get_data
from sklearn import linear_model
from sklearn import metrics
import numpy as np
from scipy.special import expit


def logistic_decode(prediction, string=False):
    prediction = expit(prediction)
    for i in xrange(len(prediction)):
        raw_score = prediction[i] * 5.0
        if raw_score < 1:
            raw_score = 1.0
        prediction[i] = round(raw_score)
        if string:
            prediction[i] = str(int(prediction[i]))
    return prediction


def raw_decode(prediction, string=False):
    for i in xrange(len(prediction)):
        raw_score = prediction[i]
        if raw_score < 1.0:
            raw_score = 1.0
        if raw_score > 5.0:
            raw_score = 5.0
        prediction[i] = round(raw_score)
        if string:
            prediction[i] = str(int(prediction[i]))

    return prediction


def regression_baseline(decode='raw', epochs=100):
    train_x, train_y, validate_x, validate_y, test_x, test_y = get_data(dim=300)

    train_x = np.asarray(train_x).astype('float64')
    train_y = np.asarray(train_y).astype('float64')

    validate_x = np.asarray(validate_x).astype('float64')
    validate_y = np.asarray(validate_y).astype('float64')

    test_x = np.asarray(test_x).astype('float64')
    test_y = np.asarray(test_y).astype('float64')

    if decode not in ['logistic', 'raw']:
        raise NotImplementedError

    regressor = linear_model.SGDRegressor(eta0=0.1, loss='squared_epsilon_insensitive', n_iter=1,
                                          warm_start=True, random_state=42)

    print "\ntraining regression model..."
    best_score = -1
    for i in xrange(epochs):
        regressor.fit(train_x, train_y)
        if decode == 'raw':
            predicted = raw_decode(regressor.predict(validate_x))
        elif decode == 'logistic':
            predicted = logistic_decode(regressor.predict(validate_x))
        score = np.mean(predicted == validate_y)
        if score > best_score:
            best_score = score
            best_regressor = regressor
        print "at epoch %d, score is %f" % (i, score)

    print "\nbest score on validation set is %f" % best_score
    print "\nfinal testing..."
    if decode == 'raw':
        predicted = raw_decode(best_regressor.predict(test_x))
    elif decode == 'logistic':
        predicted = logistic_decode(best_regressor.predict(test_x))
    print np.mean(predicted == test_y)
    print(metrics.classification_report(test_y, predicted))


if __name__ == '__main__':
    regression_baseline(decode='raw')

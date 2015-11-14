from train import get_data
import numpy as np
from sklearn import linear_model
from sklearn import metrics


def classification_baseline(api='sklearn'):
    if api == 'sklearn':
        return classification_sklearn()
    elif api == 'pylearn2':
        return classification_pylearn2()
    else:
        raise NotImplementedError


def classification_pylearn2():

    raise NotImplementedError


def classification_sklearn(epochs=100, vector='cov_bigram', kernel=(0.5, 0.5)):
    train_x, train_y, validate_x, validate_y, test_x, test_y = get_data(dim=300, vector=vector, kernel=kernel)

    train_x = np.asarray(train_x).astype('float64')
    validate_x = np.asarray(validate_x).astype('float64')
    test_x = np.asarray(test_x).astype('float64')

    classifier = linear_model.SGDClassifier(learning_rate='constant', loss='squared_hinge', eta0=0.01,
                                            penalty='l2', n_iter=1, warm_start=True, random_state=42)

    print "\ntraining regression model..."
    best_score = -1
    for i in xrange(epochs):
        classifier.fit(train_x, train_y)
        predicted = classifier.predict(validate_x)
        score = np.mean(predicted == validate_y)
        if score > best_score:
            best_score = score
            best_classifier = classifier
        print "\nat epoch %d, score is %f" % (i, score)

    print "\nbest score on validation set is %f" % best_score
    print "\nfinal testing..."
    predicted = best_classifier.predict(test_x)
    print best_classifier.score(test_x, test_y)
    print(metrics.classification_report(test_y, predicted))


if __name__ == '__main__':
    classification_baseline()
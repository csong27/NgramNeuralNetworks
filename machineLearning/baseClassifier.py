from machineLearning import get_data
import numpy as np
from sklearn import linear_model
from sklearn import metrics


def classification_baseline():
    train_x, train_y, validate_x, validate_y, test_x, test_y = get_data(dim=300)

    train_x = np.asarray(train_x).astype('float64')
    validate_x = np.asarray(validate_x).astype('float64')
    test_x = np.asarray(test_x).astype('float64')

    classifier = linear_model.SGDClassifier(loss='hinge', penalty='l2', n_iter=1)

    print "\ntraining regression model..."
    best_score = -1
    for i in xrange(50):
        classifier = classifier.fit(train_x, train_y)
        predicted = classifier.predict(validate_x)
        score = np.mean(predicted == validate_y)
        if score > best_score:
            best_score = score
            best_classifier = classifier
        print "at epoch %d, score is %f" % (i, score)

    print "\nbest score on validation set is %f" % best_score
    print "\nfinal testing..."
    predicted = best_classifier.predict(test_x)
    print best_classifier.score(test_x, test_y)
    print(metrics.classification_report(test_y, predicted))


if __name__ == '__main__':
    classification_baseline()
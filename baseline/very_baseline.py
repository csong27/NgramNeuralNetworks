__author__ = 'Song'
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from utils.load_data.load_yelp import read_train_data, read_test_data


def get_data():
    print "loading data..."
    train_x, train_y, validate_x, validate_y = read_train_data()
    train_x.extend(validate_x)
    train_y.extend(validate_y)
    test_x, test_y = read_test_data()
    return train_x, train_y, test_x, test_y


def naive_bayes():
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()), ])
    train_x, train_y, test_x, test_y = get_data()
    print "training naive bayes..."
    text_clf = text_clf.fit(train_x, train_y)
    predicted = text_clf.predict(test_x)
    print np.mean(predicted == test_y)
    print(metrics.classification_report(test_y, predicted))


def svm():
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)), ])
    train_x, train_y, test_x, test_y = get_data()
    print "training svm..."
    text_clf = text_clf.fit(train_x, train_y)
    predicted = text_clf.predict(test_x)
    print np.mean(predicted == test_y)
    print(metrics.classification_report(test_y, predicted))


if __name__ == '__main__':
    # naive_bayes()
    svm()
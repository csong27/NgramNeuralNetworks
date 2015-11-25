__author__ = 'Song'
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from path import Path
from utils.load_data import *


def get_data(data=SST_KAGGLE):
    print "loading data..."
    if data == SST_KAGGLE:
        return read_kaggle_raw(validate_ratio=0.2)


def naive_bayes():
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()), ])
    train_x, train_y, test_x, test_y = get_data()
    print "training naive bayes..."
    text_clf = text_clf.fit(train_x, train_y)
    predicted = text_clf.predict(test_x)
    print np.mean(predicted == test_y)
    print(metrics.classification_report(test_y, predicted))


def svm(epochs=10, data=SST_KAGGLE):
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=10, random_state=42,
                                               warm_start=True)), ])
    train_x, train_y, validate_x, validate_y, test_x = get_data(data=data)
    print "training svm..."
    best_score = -1
    for i in xrange(epochs):
        text_clf.fit(train_x, train_y)
        predicted = text_clf.predict(validate_x)
        score = np.mean(predicted == validate_y)
        if score > best_score:
            best_score = score
            best_classifier = text_clf
        print "\nat epoch %d, score is %f" % (i, score)

    print "\nbest score on validation set is %f" % best_score
    print "\nfinal testing..."
    best_prediction = best_classifier.predict(test_x)

    import csv
    save_path = Path('C:/Users/Song/Course/571/hw3/smv_kaggle_result.csv')
    with open(save_path, 'wb') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['PhraseId', 'Sentiment'])
        phrase_ids = np.arange(156061, 222353)
        for phrase_id, sentiment in zip(phrase_ids, best_prediction):
            writer.writerow([phrase_id, sentiment])


if __name__ == '__main__':
    # naive_bayes()
    svm()
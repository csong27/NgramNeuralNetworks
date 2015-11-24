from utils.lexicon import senti_wordnet_vectorizer, senti_lexicon_vectorizor
from utils.load_data import *
from utils import save_csv
from pickled_feature import read_probability_pickle, get_concatenated_document_vectors
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from scipy import sparse
from sklearn import linear_model, metrics
from sklearn.cross_validation import train_test_split


def classification_baseline(api='sklearn'):
    if api == 'sklearn':
        return classification_sklearn()
    elif api == 'pylearn2':
        return classification_pylearn2()
    else:
        raise NotImplementedError


def prepare_data(data=SST_KAGGLE):
    if data == SST_KAGGLE:
        train_x_1, test_x_1 = senti_lexicon_vectorizor(data=data, tfidf=True)
        train_x_2, test_x_2 = senti_wordnet_vectorizer(data=data, tfidf=False)
        train_x_3, test_x_3 = read_probability_pickle(data=data)
        train_x_4, test_x_4 = get_concatenated_document_vectors(data=data)
        _, train_y, _ = read_sst_kaggle_pickle()
        train_x = sparse.hstack((train_x_1, train_x_2, train_x_3, train_x_4))
        test_x = sparse.hstack((test_x_1, test_x_2, test_x_3, test_x_4))
        return train_x, train_y, test_x


def classification_pylearn2():
    raise NotImplementedError


def classification_sklearn(data=SST_KAGGLE, epochs=100):
    train_x, train_y, test_x = prepare_data(data=data)
    print train_x.shape
    train_x, validate_x, train_y, validate_y = train_test_split(train_x, train_y, test_size=0.2,
                                                                random_state=42, stratify=train_y)

    classifier = linear_model.SGDClassifier(loss='log', eta0=0.02, penalty='l2', n_iter=1, l1_ratio=0.00001,
                                            warm_start=True, random_state=42, learning_rate='constant')

    print "\ntraining classification model..."
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

    # classifier = RandomForestClassifier(n_estimators=50, verbose=1, n_jobs=2, warm_start=True)
    # classifier.fit(train_x, train_y)
    #
    # print classifier.score(validate_x, validate_y)

    # classifier.fit(validate_x, validate_y)

    save_csv(predicted)

if __name__ == '__main__':
    classification_baseline()
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from utils.save_kaggle_result import save_csv
from utils.lexicon import senti_lexicon_vectorizor, senti_wordnet_vectorizer
from utils.load_data import SST_KAGGLE, read_sst_kaggle_pickle
from bag_of_bigram import vectorize_text
from scipy import sparse
from doc_embedding import read_aggregated_vectors
import numpy as np
import cPickle as pkl
ALGS = ['svm', 'log', 'rig', 'per']
DATA = ['bow', 'average']


def average_word_kaggle_dataset():
    return read_aggregated_vectors(data=SST_KAGGLE, google=True)


def bow_kaggle_dataset():
    train_x, train_y, test_x = vectorize_text(data=SST_KAGGLE, tfidf=True)
    train_x_s, train_y, test_x_s = sentiment_kaggle_dataset()
    train_x = sparse.hstack([train_x, train_x_s])
    test_x = sparse.hstack([test_x, test_x_s])
    return train_x, train_y, test_x


def sentiment_kaggle_dataset():
    train_x_1, test_x_1 = senti_lexicon_vectorizor(data=SST_KAGGLE, tfidf=True)
    train_x_2, test_x_2 = senti_wordnet_vectorizer(data=SST_KAGGLE, tfidf=True)
    train_x = sparse.hstack((train_x_1, train_x_2))
    test_x = sparse.hstack((test_x_1, test_x_2))
    _, train_y, _ = read_sst_kaggle_pickle()
    return train_x, train_y, test_x


def train_kaggle(dataset, alg='rig'):
    train_x, train_y, test_x = dataset
    print "shape for training data is", train_x.shape

    if alg == 'svm':
        clf = SGDClassifier(verbose=1, n_jobs=2, n_iter=25)
    elif alg == 'log':
        clf = LogisticRegression(verbose=1, n_jobs=2)
    elif alg == 'per':
        clf = Perceptron(verbose=1, n_jobs=2, n_iter=25)
    elif alg == 'rig':
        clf = RidgeClassifier()
    else:
        raise NotImplementedError

    print "training with %s..." % alg

    clf.fit(train_x, train_y)
    # clf.fit(validate_x, validate_y)
    predicted = clf.predict(test_x)
    save_csv(predicted, fname=alg)

    return clf.decision_function(train_x), clf.decision_function(test_x)


def save_predict_score(data='bow', alg='log'):
    path = 'D:/data/nlpdata/pickled_data/score/'
    if data == 'bow':
        dataset = bow_kaggle_dataset()
    elif data == 'average':
        dataset = average_word_kaggle_dataset()
    else:
        dataset = sentiment_kaggle_dataset()
    train_x_prob, test_x_prob = train_kaggle(dataset=dataset, alg=alg)
    path += data + '_' + alg + '.pkl'
    f = open(path, 'wb')
    pkl.dump((train_x_prob, test_x_prob), f, -1)
    f.close()


def read_predict_score(data='bow', alg='rig'):
    path = 'D:/data/nlpdata/pickled_data/score/'
    path += data + '_' + alg + '.pkl'
    f = open(path, 'rb')
    train_x_prob, test_x_prob = pkl.load(f)
    return train_x_prob, test_x_prob


def save_wrapper(data='bow'):
    for alg in ALGS:
        save_predict_score(data=data, alg=alg)


def read_all_predict_score(data='bow'):
    train_x = []
    test_x = []
    for alg in ALGS:
        train_x_prob, test_x_prob = read_predict_score(data=data, alg=alg)
        train_x.append(train_x_prob)
        test_x.append(test_x_prob)
    train_x = np.concatenate(train_x, axis=1)
    test_x = np.concatenate(test_x, axis=1)
    return train_x, test_x


def multi_learner():
    train_x_2, train_y, test_x_2 = read_aggregated_vectors()

    train_x = [train_x_2]
    test_x = [test_x_2]
    for alg in ALGS:
        train_x_prob, test_x_prob = read_predict_score(alg=alg)
        train_x.append(train_x_prob)
        test_x.append(test_x_prob)

    train_x = np.concatenate(train_x, axis=1)
    test_x = np.concatenate(test_x, axis=1)

    print "training..."
    clf = LogisticRegression(n_jobs=2)
    clf.fit(train_x, train_y)
    predicted = clf.predict(test_x)
    save_csv(predicted)


if __name__ == '__main__':
    multi_learner()

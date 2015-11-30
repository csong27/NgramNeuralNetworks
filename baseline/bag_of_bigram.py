from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from utils.save_kaggle_result import save_csv
from utils.load_data import *
from utils.lexicon import senti_lexicon_vectorizor, senti_wordnet_vectorizer
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from doc_embedding import read_doc2vec_pickle
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from gensim.models import Phrases


def to_phrase():
    train_x, train_y, test_x = read_sst_kaggle_pickle(use_textblob=False)
    phrase = Phrases(train_x)
    for i, x in enumerate(train_x):
        train_x[i] = " ".join(phrase[train_x[i]])
    for i, x in enumerate(test_x):
        test_x[i] = " ".join(phrase[test_x[i]])
    return train_x, train_y, test_x


def vectorize_text(data=SST_KAGGLE, tfidf=True):
    if tfidf:
        bigram_vectorizer = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', stop_words='english', min_df=2)
    else:
        bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', stop_words='english', min_df=2)
    if data == SST_KAGGLE:
        train_x, train_y, test_x = read_kaggle_raw()
    else:
        raise NotImplementedError
    train_x = bigram_vectorizer.fit_transform(train_x)
    train_y = np.asarray(train_y)
    test_x = bigram_vectorizer.transform(test_x)
    return train_x, train_y, test_x


def train(data=SST_KAGGLE, alg='nb'):
    train_x, train_y, test_x = vectorize_text(data=data)
    # train_x_1, test_x_1 = senti_lexicon_vectorizor(data=data, tfidf=True)
    # train_x_2, test_x_2 = senti_wordnet_vectorizer(data=data, tfidf=True)
    #
    # train_x = sparse.hstack((train_x_1, train_x_2))
    # test_x = sparse.hstack((test_x_1, test_x_2))

    print "shape for training data is", train_x.shape

    if alg == 'svm':
        clf = SVC(verbose=1)
    elif alg == 'log':
        clf = LogisticRegression(verbose=1)     # 61.756, no phrase,
    elif alg == 'nb':
        clf = MultinomialNB()
    else:
        raise NotImplementedError

    print "training..."
    clf.fit(train_x, train_y)
    predicted = clf.predict(test_x)
    save_csv(predicted)

if __name__ == '__main__':
    train()

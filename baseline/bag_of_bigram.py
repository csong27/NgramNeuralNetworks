from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from utils.save_kaggle_result import save_csv
from utils.load_data import *
from utils.lexicon import senti_lexicon_vectorizor, senti_wordnet_vectorizer
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB


def vectorize_text(data=SST_KAGGLE, tfidf=False):
    if tfidf:
        bigram_vectorizer = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', stop_words='english')
    else:
        bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', stop_words='english')
    if data == SST_KAGGLE:
        train_x, train_y, test_x = read_kaggle_raw()
    else:
        raise NotImplementedError
    train_x = bigram_vectorizer.fit_transform(train_x)
    train_y = np.asarray(train_y)
    test_x = bigram_vectorizer.transform(test_x)
    return train_x, train_y, test_x


def train(data=SST_KAGGLE, alg='svm'):
    train_x, train_y, test_x = vectorize_text(data=data)
    train_x_1, test_x_1 = senti_lexicon_vectorizor(data=data, tfidf=True)
    train_x_2, test_x_2 = senti_wordnet_vectorizer(data=data, tfidf=True)

    train_x = sparse.hstack((train_x_1, train_x_2, train_x))
    test_x = sparse.hstack((test_x_1, test_x_2, test_x))

    if alg == 'svm':
        clf = SVC(verbose=1)
    elif alg == 'log':
        clf = LogisticRegression(verbose=1)     # 61.756
    elif alg == 'nb':
        clf = MultinomialNB()
    else:
        raise NotImplementedError

    print "training..."
    clf.fit(train_x, train_y)
    # clf.fit(validate_x, validate_y)
    predicted = clf.predict(test_x)
    save_csv(predicted)

if __name__ == '__main__':
    train()

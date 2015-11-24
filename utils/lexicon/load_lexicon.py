from path import Path
from utils.load_data import *
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


pos_path = Path("D:/data/nlpdata/lexicon/positive-words.txt")
neg_path = Path("D:/data/nlpdata/lexicon/negative-words.txt")
subj_path = Path("D:/data/nlpdata/lexicon/subjclueslen.tff")
inquire_path = Path("D:/data/nlpdata/lexicon/inquirerbasic.txt")


def read_subj_lexicon(p=subj_path):
    f = open(p)
    words = set()
    for line in f:
        arr = line.split(' ')
        word = arr[2].split('=')[1]
        words.add(word)
    return words


def read_lexicon(p):
    f = open(p)
    words = set()
    for line in f:
        if line[0] == ';':
            continue
        if line[-1] == '\n':
            line = line[:-1]
        words.add(line)
    return words


def read_inquire(p=inquire_path):
    f = open(p)
    f.readline()
    words = set()
    for line in f:
        arr = line.split('\t')
        word = arr[0].lower().split('#')[0]
        pos = arr[2]
        neg = arr[3]
        if pos != '' or neg != '':
            words.add(word)
    return words


def senti_lexicon_vectorizor(data=SST_KAGGLE, tfidf=True):
    if data == SST_KAGGLE:
        train_x, train_y, test_x = read_sst_kaggle_pickle()
        train_words = train_x + test_x
        train_words = set([word for sentence in train_words for word in sentence])
    else:
        raise NotImplementedError
    pos_words = read_lexicon(pos_path)
    neg_words = read_lexicon(neg_path)
    subj_words = read_subj_lexicon()
    inquire_words = read_inquire()
    words = pos_words.union(neg_words, subj_words, inquire_words)

    vocab = train_words.intersection(words)

    all_x = train_x + test_x
    input_sent = []
    for sent in all_x:
        sent_dict = defaultdict(int)
        for word in sent:
            if word in vocab:
                sent_dict[word] += 1  # add word sentiment score
        input_sent.append(sent_dict)
    # dict vectorization
    dv = DictVectorizer()
    all_x = dv.fit_transform(input_sent)

    if tfidf:
        # tf-idf vectorization
        tv = TfidfTransformer()
        all_x = tv.fit_transform(all_x)

    # get train and test data
    train_len = len(train_x)
    train_x = all_x[:train_len]
    test_x = all_x[train_len:]

    return train_x, test_x


if __name__ == '__main__':
    senti_lexicon_vectorizor()

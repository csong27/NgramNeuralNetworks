from path import Path
from utils.load_data import *
from utils.preprocess import STOPWORDS
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

wordnet_path = Path("D:/data/nlpdata/lexicon/SentiWordNet.txt")


def add_word(words, word, scores):
    if "_" in word:
        return
    word = word.split("#")[0]
    words[word] = scores


def read_wordnet(p=wordnet_path):
    f = open(p)
    word_dict = dict()
    for line in f:
        if line[0] == '#':
            continue
        data = line.split("\t")
        pos_score = float(data[2])
        neg_score = float(data[3])
        words = data[4]
        if pos_score != 0 or neg_score != 0:
            scores = (pos_score, neg_score)
            arr = words.split(" ")
            if len(arr) > 1:
                for word in arr:
                    add_word(word_dict, word, scores)
            else:
                add_word(word_dict, words, scores)

    return word_dict


def senti_wordnet_vectorizer(data=SST_KAGGLE, tfidf=True):
    print "loading sentiment wordnet"
    if data == SST_KAGGLE:
        train_x, train_y, test_x = read_sst_kaggle_pickle()
        train_words = train_x + test_x
        train_words = set([word for sentence in train_words for word in sentence])
    else:
        raise NotImplementedError
    word_dict = read_wordnet()

    vocab = train_words.intersection(word_dict.keys())

    # preprocess vocab
    for word in vocab:
        if word in STOPWORDS:
            del word_dict[word]

    vocab = train_words.intersection(word_dict.keys())

    input_sent = []  # input to dict vectorizer
    all_x = train_x + test_x
    for sent in all_x:
        sent_dict = defaultdict(float)
        for word in sent:
            if word in vocab:   # add word sentiment score
                sent_dict[word + "_pos"] += word_dict[word][0]
                sent_dict[word + "_neg"] += word_dict[word][1]

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
    senti_wordnet_vectorizer()

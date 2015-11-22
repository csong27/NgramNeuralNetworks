from path import Path
from utils.load_data import *
from collections import OrderedDict
import numpy as np
from scipy.sparse import csr_matrix


wordnet_path = Path("D:/data/nlpdata/lexicon/SentiWordNet.txt")


def add_word(words, word, score):
    if "_" in word:
        return
    word = word.split("#")[0]
    words[word] = score


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
            if pos_score != 0 and neg_score != 0:
                score = 1 - (pos_score + neg_score)
            elif pos_score != 0:
                score = 1 - pos_score
            else:
                score = 1 - neg_score
            if score == 0:
                continue
            arr = words.split(" ")
            if len(arr) > 1:
                for word in arr:
                    add_word(word_dict, word, score)
            else:
                add_word(word_dict, words, score)

    return word_dict


def get_sentiment_worddict(data=SST_KAGGLE):
    if data == SST_KAGGLE:
        train_x, train_y, test_x = read_sst_kaggle_pickle()
        train_words = train_x + test_x
        train_words = set([word for sentence in train_words for word in sentence])
    else:
        raise NotImplementedError
    word_dict = read_wordnet()

    words = word_dict.keys()
    words = train_words.intersection(words)
    words = sorted(words)

    vectors = []
    for sent in train_x:
        vector_dict = OrderedDict().fromkeys(words, 0)
        for word in sent:
            if word in words:
                vector_dict[word] += word_dict[word]
        vectors.append(np.asarray(vector_dict.values(), dtype='float32'))
    vectors = np.asarray(vectors)
    spr_vectors = csr_matrix(vectors)
    print spr_vectors

if __name__ == '__main__':
    get_sentiment_worddict()
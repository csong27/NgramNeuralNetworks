from path import Path
from utils.load_data import *


pos_path = Path("D:/data/nlpdata/lexicon/positive-words.txt")
neg_path = Path("D:/data/nlpdata/lexicon/negative-words.txt")


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


def get_sentiment_wordset(data=SST_KAGGLE):
    if data == SST_KAGGLE:
        train_x, train_y, test_x = read_sst_kaggle_pickle()
        train_words = train_x + test_x
        train_words = set([word for sentence in train_words for word in sentence])
        print train_words
    else:
        raise NotImplementedError
    pos_words = read_lexicon(pos_path)
    neg_words = read_lexicon(neg_path)
    words = pos_words.union(neg_words)

    words = train_words.intersection(words)

    x = train_x + test_x
    count = 0
    for sent in x:
        sent_count = 0
        for word in sent:
            if word in words:
                sent_count += 1
        if sent_count == 0:
            count += 1
    print count


if __name__ == '__main__':
    get_sentiment_wordset()
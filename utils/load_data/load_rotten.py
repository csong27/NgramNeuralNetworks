from path import Path
from utils.preprocess import SIMPLE_FILTERS, preprocess_review
import cPickle as pkl
import platform
if platform.system() == 'Windows':
    rotten_pickle = Path('C:/Users/Song/Course/571/project/pickled_data/rt.pkl')
    pos = Path('D:/data/nlpdata/rotten/rt-polarity.pos')
    neg = Path('D:/data/nlpdata/rotten/rt-polarity.neg')
else:
    rotten_pickle = Path('/home/scz8928999/data/pickled/rt.pkl')
    pos = Path('/home/scz8928999/data/rotten/rt-polarity.pos')
    neg = Path('/home/scz8928999/data/rotten/rt-polarity.neg')


def read_data(p):
    label = 1 if p[-3:] == 'pos' else 0
    f = open(p)
    x = []
    for line in f:
        line = unicode(line, errors='ignore')
        x.append(preprocess_review(line, filters=SIMPLE_FILTERS))
    y = [label] * len(x)
    return x, y


def save_rotten_pickle():
    pos_x, pos_y = read_data(pos)
    neg_x, neg_y = read_data(neg)
    x = pos_x + neg_x
    y = pos_y + neg_y
    f = open('rt.pkl', 'wb')
    pkl.dump((x, y), f, -1)
    f.close()


def read_rotten_pickle():
    f = open(rotten_pickle, 'rb')
    x, y = pkl.load(f)
    return x, y


if __name__ == '__main__':
    save_rotten_pickle()
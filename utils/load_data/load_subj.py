from path import Path
from utils.preprocess import SIMPLE_FILTERS, preprocess_review
import cPickle as pkl

import platform

if platform.system() == 'Windows':
    subj_pickle = Path('C:/Users/Song/Course/571/project/pickled_data/subj.pkl')
    subj = Path('D:/data/nlpdata/subj/subjective.pos')
    obj = Path('D:/data/nlpdata/subj/objective.neg')
else:
    subj_pickle = Path('/home/scz8928999/data/pickled/subj.pkl')
    subj = Path('/home/scz8928999/data/subj/subjective.pos')
    obj = Path('/home/scz8928999/data/subj/objective.neg')


def read_data(p):
    label = 1 if p[-3:] == 'pos' else 0
    f = open(p)
    x = []
    for line in f:
        line = unicode(line, errors='ignore')
        x.append(preprocess_review(line, filters=SIMPLE_FILTERS))
    y = [label] * len(x)
    return x, y


def save_subj_pickle():
    pos_x, pos_y = read_data(subj)
    neg_x, neg_y = read_data(obj)
    print pos_y[0]
    print neg_y[0]
    x = pos_x + neg_x
    y = pos_y + neg_y
    f = open('subj.pkl', 'wb')
    pkl.dump((x, y), f, -1)
    f.close()


def read_subj_pickle():
    f = open(subj_pickle, 'rb')
    x, y = pkl.load(f)
    return x, y
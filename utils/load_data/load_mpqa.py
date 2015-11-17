from path import Path
import cPickle as pkl

mpqa_pickle = Path('C:/Users/Song/Course/571/project/pickled_data/mpqa.pkl')
pos = Path('D:/data/nlpdata/mpqa/mpqa.pos')
neg = Path('D:/data/nlpdata/mpqa/mpqa.neg')


def read_data(p):
    label = 1 if p[-3:] == 'pos' else 0
    f = open(p)
    x = []
    for line in f:
        line = unicode(line, errors='ignore')
        if line[-1] == '\n':
            line = line[:-1]
        x.append(line.split(' '))
    y = [label] * len(x)
    return x, y


def save_mpqa_pickle():
    pos_x, pos_y = read_data(pos)
    neg_x, neg_y = read_data(neg)
    x = pos_x + neg_x
    y = pos_y + neg_y
    f = open('mpqa.pkl', 'wb')
    pkl.dump((x, y), f, -1)
    f.close()


def read_mpqa_pickle():
    f = open(mpqa_pickle, 'rb')
    x, y = pkl.load(f)
    return x, y
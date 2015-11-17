from path import Path
from sklearn.cross_validation import train_test_split
import numpy as np
import cPickle as pkl

LABEL = {'DESC': 0, 'HUM': 1, 'ENTY': 2, 'ABBR': 3, 'NUM': 4, 'LOC': 5}

trec_pickle = Path('C:/Users/Song/Course/571/project/pickled_data/trec.pkl')
train_path = Path('D:/data/nlpdata/trec/train_5500.label')
test_path = Path('D:/data/nlpdata/trec/TREC_10.label')


def read_data(p):
    f = open(p)
    x = []
    y = []
    for line in f:
        line = unicode(line, errors='ignore')
        line = line.lower()
        data = line.split(' ')
        label = data[0].split(':')[0].upper()
        y.append(LABEL[label])
        x.append(data[1:-1])
    return x, y


def save_trec_pickle(validate_ratio=0.1):
    train_x, train_y = read_data(train_path)
    train_y = np.asarray(train_y)
    test_x, test_y = read_data(test_path)
    train_x, validate_x, train_y, validate_y = train_test_split(train_x, train_y, test_size=validate_ratio,
                                                                random_state=42, stratify=train_y)
    f = open('trec.pkl', 'wb')

    pkl.dump((train_x, train_y), f, -1)
    pkl.dump((validate_x, validate_y), f, -1)
    pkl.dump((test_x, test_y), f, -1)
    f.close()


def read_trec_pickle():
    f = open(trec_pickle, 'rb')
    train_x, train_y = pkl.load(f)
    validate_x, validate_y = pkl.load(f)
    test_x, test_y = pkl.load(f)
    return train_x, train_y, validate_x, validate_y, test_x, test_y
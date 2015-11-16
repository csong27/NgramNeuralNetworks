dataset_path = 'D:/data/nlpdata/aclImdb/'
from preprocess import preprocess_review
import numpy as np
import cPickle as pkl
import glob
import os


def read_imdb_pickle(path='imdb.pkl', valid_portion=0.1):
    f = open(path, 'rb')
    train_x, train_y = pkl.load(f)
    test_x, test_y = pkl.load(f)

    n_samples = len(train_x)
    sidx = np.random.permutation(n_samples)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_x = [train_x[s] for s in sidx[n_train:]]
    valid_y = [train_y[s] for s in sidx[n_train:]]
    train_x = [train_x[s] for s in sidx[:n_train]]
    train_y = [train_y[s] for s in sidx[:n_train]]

    print len(train_x)
    print len(valid_x)


def grab_data(path):
    sentences = []
    currdir = os.getcwd()
    os.chdir(path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(preprocess_review(f.readline().strip()))
    os.chdir(currdir)
    return sentences


def save_imdb_pickle():
    path = dataset_path
    train_x_pos = grab_data(path + 'train/pos')
    train_x_neg = grab_data(path+'train/neg')
    train_x = train_x_pos + train_x_neg
    train_y = [1] * len(train_x_pos) + [0] * len(train_x_neg)

    test_x_pos = grab_data(path+'test/pos')
    test_x_neg = grab_data(path+'test/neg')
    test_x = test_x_pos + test_x_neg
    test_y = [1] * len(test_x_pos) + [0] * len(test_x_neg)

    f = open('imdb.pkl', 'wb')
    pkl.dump((train_x, train_y), f, -1)
    pkl.dump((test_x, test_y), f, -1)
    f.close()

if __name__ == '__main__':
    read_imdb_pickle()

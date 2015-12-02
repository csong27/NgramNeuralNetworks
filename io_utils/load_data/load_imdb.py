from io_utils.preprocess import preprocess_review
from sklearn.cross_validation import train_test_split
from path import Path
import cPickle as pkl
import glob
import os

dataset_path = 'D:/data/nlpdata/aclImdb/'
imdb_pickle = Path('C:/Users/Song/Course/571/project/pickled_data/imdb.pkl')


def read_imdb_pickle():
    f = open(imdb_pickle, 'rb')
    train_x, train_y = pkl.load(f)
    test_x, test_y = pkl.load(f)
    return train_x, train_y, test_x, test_y


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
    save_imdb_pickle()

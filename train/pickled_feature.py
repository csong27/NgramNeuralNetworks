from doc_embedding import *
from utils.load_data import *
from path import Path
import cPickle as pkl
import numpy as np


def read_probability_pickle(data=SST_KAGGLE):
    save_path = "D:/data/nlpdata/pickled_data/" + SST_KAGGLE + "_prob.pkl"
    print "reading probability feature from %s" % save_path

    f = open(Path(save_path), "rb")
    train_prob, test_prob = pkl.load(f)
    f.close()
    return train_prob, test_prob


def get_concatenated_document_vectors(data=SST_KAGGLE):
    train_x_1, test_x_1 = read_doc2vec_pickle(dm=True, concat=False, data=data)
    train_x_2, test_x_2 = read_doc2vec_pickle(dm=False, concat=False, data=data)
    train_x_3, test_x_3 = read_ngram_vectors(data=data)

    train_x = np.concatenate((train_x_1, train_x_2, train_x_3), axis=1)
    test_x = np.concatenate((test_x_1, test_x_2, test_x_3), axis=1)

    return train_x, test_x


def read_ngram_vectors(data=SST_KAGGLE):
    save_path = "D:/data/nlpdata/pickled_data/doc2vec/"
    save_path += data + "_ngram.pkl"
    print "reading doc2vec from %s" % save_path

    f = open(Path(save_path), "rb")
    saved_train, saved_test = pkl.load(f)
    f.close()

    return saved_train, saved_test
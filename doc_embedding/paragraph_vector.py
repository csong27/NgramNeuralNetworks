from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from utils.load_data import *
from random import shuffle
from copy import copy
from path import Path
import cPickle as pkl
import numpy as np
import multiprocessing
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
cores = multiprocessing.cpu_count()


def get_tagged_document(x, label):
    documents = []
    for index, sent in enumerate(x):
        tag = "%s;%d" % (label, index)
        document = TaggedDocument(words=sent, tags=[tag])
        documents.append(document)
    return documents


def get_vectors(model, docs, trained=True):
    vectors = []
    for doc in docs:
        if trained:
            vectors.append(model.docvecs[doc.tags[0]])
        else:
            vectors.append(model.infer_vector(doc))
    return np.asarray(vectors)


def train_doc2vec(data, dm=True, concat=False, negative=10, size=100, epochs=50, alpha=0.025, min_alpha=0.001, sample=1e-4):
    if data == SST_KAGGLE:
        train_x, train_y, test_x = read_sst_kaggle_pickle()
    else:
        return
    train_doc = get_tagged_document(train_x, 'train')
    test_doc = get_tagged_document(test_x, 'test')

    if dm and concat:
        model = Doc2Vec(dm=1, dm_concat=1, size=size, window=5, negative=negative, hs=0, min_count=1, sample=sample, workers=cores)
    elif dm:
        model = Doc2Vec(dm=1, dm_mean=1, size=size, window=10, negative=negative, hs=0, min_count=1, sample=sample, workers=cores)
    else:
        model = Doc2Vec(dm=0, size=size, negative=negative, window=10, hs=0, min_count=1, sample=sample, workers=cores)
    all_doc = train_doc + test_doc
    model.build_vocab(all_doc)

    alpha_delta = (alpha - min_alpha) / epochs
    for epoch in range(epochs):
        np.random.shuffle(all_doc)
        model.alpha, model.min_alpha = alpha, alpha
        model.train(all_doc)
        alpha -= alpha_delta
    print "\ngetting trained vectors..."
    train_x = get_vectors(model, train_doc)
    test_x = get_vectors(model, test_x, trained=False)

    return train_x, test_x


def save_doc2vec_pickle(dm=True, concat=True, size=200, data=SST_KAGGLE):
    save_path = "D:/data/nlpdata/pickled_data/doc2vec/"
    save_path += data
    if dm and concat:
        model_name = "_dm_concat.pkl"
    elif dm:
        model_name = "_dm.pkl"
    else:
        model_name = "_dbow.pkl"
    save_path += model_name
    train_x, test_x = train_doc2vec(data=SST_KAGGLE, dm=dm, concat=concat, size=size)

    print "saving doc2vec to %s" % save_path

    f = open(Path(save_path), "wb")
    pkl.dump((train_x, test_x), f, -1)
    f.close()


def read_doc2vec_pickle(dm=True, concat=True, data=SST_KAGGLE):
    save_path = "D:/data/nlpdata/pickled_data/doc2vec/"
    save_path += data
    if dm and concat:
        model_name = "_dm_concat.pkl"
    elif dm:
        model_name = "_dm.pkl"
    else:
        model_name = "_dbow.pkl"
    save_path += model_name

    print "reading doc2vec from %s" % save_path

    f = open(Path(save_path), "rb")
    train_x, test_x = pkl.load(f)
    f.close()
    return train_x, test_x


if __name__ == '__main__':
    save_doc2vec_pickle(dm=True, concat=False, size=100)
    save_doc2vec_pickle(dm=False, concat=False, size=100)

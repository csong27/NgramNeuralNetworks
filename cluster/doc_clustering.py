from utils.pickled_feature import read_ngram_vectors
from doc_embedding.paragraph_vector import read_doc2vec_pickle
from utils.load_data import *
from sklearn.cluster import KMeans
import numpy as np


def cluster_document(data=SST_KAGGLE):
    train_doc, _, test_doc = read_sst_kaggle_pickle(use_textblob=True)
    all_doc = train_doc + test_doc

    train_x_1, test_x_1 = read_doc2vec_pickle(dm=True, concat=False, data=data)
    train_x_2, test_x_2 = read_doc2vec_pickle(dm=False, concat=False, data=data)
    train_x_3, test_x_3 = read_ngram_vectors(data=data)

    all_x_1 = np.append(train_x_1, test_x_1, axis=0)
    all_x_2 = np.append(train_x_2, test_x_2, axis=0)

    ngram_x = np.append(train_x_3, test_x_3, axis=0)
    doc2vec_x = np.concatenate((all_x_1, all_x_2), axis=1)
    all_x = np.concatenate((doc2vec_x, ngram_x), axis=1)

    kmeans = KMeans(n_clusters=50, random_state=42, verbose=1)
    idx = kmeans.fit_predict(doc2vec_x)

    print idx.shape

if __name__ == '__main__':
    cluster_document()
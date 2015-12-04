from path import Path
from gensim.models import Word2Vec
import numpy as np
import platform
from gensim import utils


if platform.system() == 'Windows':
    data_path = 'D:/data/nlpdata/'
else:
    data_path = '/home/scz8928999/vector/'

google_vector = Path(data_path + 'GoogleNews-vectors-negative300.bin')
glove_vector_50 = Path(data_path + 'glove.6B.50d.txt')
glove_vector_100 = Path(data_path + 'glove.6B.100d.txt')
glove_vector_200 = Path(data_path + 'glove.6B.200d.txt')
glove_vector_300 = Path(data_path + 'glove.6B.300d.txt')
glove_vector_huge = Path(data_path + 'glove.840B.300d.txt')


def read_google_model():
    print "reading google word embedding vectors..."
    return Word2Vec.load_word2vec_format(google_vector, binary=True)  # C binary format


def read_glove_model(dim=50, huge=False):
    print "reading gloVe word embedding vectors..."
    if dim == 50:
        return Word2Vec.load_word2vec_format(glove_vector_50, binary=False)
    elif dim == 100:
        return Word2Vec.load_word2vec_format(glove_vector_100, binary=False)
    elif dim == 200:
        return Word2Vec.load_word2vec_format(glove_vector_200, binary=False)
    elif dim == 300:
        return Word2Vec.load_word2vec_format(glove_vector_300, binary=False)
    elif huge:
        return read_glove_to_dict(glove_vector_huge)


def read_glove_to_dict(p):
    f = open(p)
    model = {}
    count = 0
    for line in f:
        arr = utils.to_unicode(line.rstrip(), encoding='utf8').split(" ")
        word = arr[0]
        vector = np.asarray(arr[1:], dtype=float)
        model[word] = vector
        count += 1
    print "loaded with %d words" % count
    return model

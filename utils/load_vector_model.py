from path import Path
from gensim.models import Word2Vec

import platform

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


def read_glove_model(dim=50):
    print "reading gloVe word embedding vectors..."
    if dim == 50:
        return Word2Vec.load_word2vec_format(glove_vector_50, binary=False)
    elif dim == 100:
        return Word2Vec.load_word2vec_format(glove_vector_100, binary=False)
    elif dim == 200:
        return Word2Vec.load_word2vec_format(glove_vector_200, binary=False)
    elif dim == 300:
        return Word2Vec.load_word2vec_format(glove_vector_300, binary=False)
    else:
        raise NotImplementedError
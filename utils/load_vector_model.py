from path import Path
from gensim.models import Word2Vec

google_vector = Path('D:/data/nlpdata/GoogleNews-vectors-negative300.bin')
glove_vector_50 = Path('D:/data/nlpdata/glove.6B.50d.txt')
glove_vector_100 = Path('D:/data/nlpdata/glove.6B.100d.txt')
glove_vector_200 = Path('D:/data/nlpdata/glove.6B.200d.txt')
glove_vector_300 = Path('D:/data/nlpdata/glove.6B.300d.txt')
glove_vector_huge = Path('D:/data/nlpdata/glove.840B.300d.txt')


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
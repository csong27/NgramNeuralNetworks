from path import Path
from gensim.models import Word2Vec

google_vector = Path('D:/data/nlpdata/GoogleNews-vectors-negative300.bin')
glove_vector_50 = Path('D:/data/nlpdata/glove.6B.50d.txt')
glove_vector_100 = Path('D:/data/nlpdata/glove.6B.100d.txt')
glove_vector_200 = Path('D:/data/nlpdata/glove.6B.200d.txt')
glove_vector_300 = Path('D:/data/nlpdata/glove.6B.300d.txt')


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


def pre_process():
    files = [(50, glove_vector_50), (100, glove_vector_100), (200, glove_vector_200), (300, glove_vector_300)]
    for item in files:
        with open(item[1], "r+") as f:
            s = f.read()
            f.seek(0)
            f.write("400000 " + str(item[0]) + "\n" + s)


if __name__ == '__main__':
    model = read_glove_model()
    print "asdaweqvsd" in model
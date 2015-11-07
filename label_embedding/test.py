__author__ = 'Song'

from my_word2vec import Word2Vec

if __name__ == '__main__':
    sentences = [['first', 'sentence'], ['second', 'sentence']]
    model = Word2Vec(sentences, min_count=1, hs=0, negative=1, sg=0)
    print model['first']
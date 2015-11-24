import logging
from utils.load_data import *
from gensim import corpora, models, utils
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def get_documents(data):
    if data == SST_KAGGLE:
        train_x, _, test_x = read_sst_kaggle_pickle(use_textblob=True)
        document = train_x + test_x
        return document
    else:
        raise NotImplementedError


class MyCorpus(object):
    def __init__(self, documents):
        self.documents = documents
        self.dictionary = corpora.Dictionary(documents)
        self.dictionary.filter_extremes(no_below=2, no_above=0.5)

    def __iter__(self):
        for tokens in self.documents:
            yield self.dictionary.doc2bow(tokens)

if __name__ == '__main__':
    kaggle_documents = get_documents(data=SST_KAGGLE)
    corpus = MyCorpus(documents=kaggle_documents)
    lda = models.LdaMulticore(corpus, id2word=corpus.dictionary, num_topics=30, workers=2, chunksize=10000,
                              iterations=100)
    print kaggle_documents[0]
    print lda[corpus.dictionary.doc2bow(kaggle_documents[0])]
    print '\n***********************\n'
    lda.print_topics(num_topics=30, num_words=15)

import logging
from utils.load_data import *
from path import Path
from gensim import corpora, models
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
lda_pickled_path = "D:/data/nlpdata/pickled_data/lda_"


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
        self.dictionary.filter_extremes(no_below=2, no_above=0.75)

    def __iter__(self):
        for tokens in self.documents:
            yield self.dictionary.doc2bow(tokens)


def train_lda(data=SST_KAGGLE, save_model=True):
    documents = get_documents(data=data)
    corpus = MyCorpus(documents=documents)
    lda = models.LdaMulticore(corpus, id2word=corpus.dictionary, num_topics=30, workers=2, chunksize=10000,
                              iterations=100)
    print documents[0]
    print lda[corpus.dictionary.doc2bow(documents[0])]
    print '\n***********************\n'
    lda.print_topics(num_topics=30, num_words=15)
    if save_model:
        fname = Path(lda_pickled_path + data + ".pkl")
        lda.save(fname=fname)


def load_lda(data=SST_KAGGLE):
    fname = lda_pickled_path + data + ".pkl"
    lda = models.LdaMulticore.load(Path(fname))
    return lda


if __name__ == '__main__':
    train_lda()

import logging
from corpus import get_documents, MyCorpus
from utils.load_data import *
from path import Path
from gensim import models

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
lda_pickled_path = "D:/data/nlpdata/pickled_data/topic/lsi_"


def train_lsi(data=SST_KAGGLE, num_topics=100, save_model=True):
    documents = get_documents(data=data)
    corpus = MyCorpus(documents=documents)
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lda = models.LsiModel(corpus_tfidf, id2word=corpus.dictionary, num_topics=num_topics, chunksize=10000)
    if save_model:
        fname = Path(lda_pickled_path + data + ".pkl")
        lda.save(fname=fname)


def load_lsi(data=SST_KAGGLE):
    fname = lda_pickled_path + data + ".pkl"
    lda = models.LsiModel.load(Path(fname))
    lda.print_topics(30)
    return lda


if __name__ == '__main__':
    load_lsi()

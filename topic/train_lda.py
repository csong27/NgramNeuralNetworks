import logging
from corpus import get_documents, MyCorpus
from io_utils.load_data import *
from path import Path
from gensim import models
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
lda_pickled_path = "D:/data/nlpdata/pickled_data/topic/lda_"


def train_lda(data=SST_KAGGLE, num_topics=30, save_model=True):
    documents = get_documents(data=data)
    corpus = MyCorpus(documents=documents)
    lda = models.LdaMulticore(corpus, id2word=corpus.dictionary, num_topics=num_topics, workers=2, chunksize=10000,
                              iterations=100)
    if save_model:
        fname = Path(lda_pickled_path + data + ".pkl")
        lda.save(fname=fname)


def load_lda(data=SST_KAGGLE):
    fname = lda_pickled_path + data + ".pkl"
    lda = models.LdaMulticore.load(Path(fname))
    lda.print_topics(50)
    return lda


if __name__ == '__main__':
    load_lda()

from utils.load_data import *
from gensim import corpora


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

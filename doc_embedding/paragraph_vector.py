from gensim.models import Doc2Vec
from utils.preprocess import MyDocuments
from utils import yelp_2013_train, yelp_2013_test
import multiprocessing

cores = multiprocessing.cpu_count()


def train_doc2vec(dm=True, concat=True, negative=20, size=100):
    documents = MyDocuments(filename=yelp_2013_train, int_label=False, str_label=False)
    if dm and concat:
        # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
        model = Doc2Vec(dm=1, dm_concat=1, size=size, window=5, negative=negative, hs=0, min_count=2, workers=cores)
    elif dm:
        # PV-DM w/average
        model = Doc2Vec(dm=1, dm_mean=1, size=size, window=10, negative=negative, hs=0, min_count=2, workers=cores)
    else:
        # PV-DBOW
        model = Doc2Vec(dm=0, size=size, negative=negative, hs=0, min_count=2, workers=cores)

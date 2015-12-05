from load_rotten import read_rotten_pickle
from load_cr import read_cr_pickle
from load_imdb import read_imdb_pickle
from load_mpqa import read_mpqa_pickle
from load_sst import read_sst_sent_pickle, read_sst_kaggle_pickle, read_kaggle_raw
from load_subj import read_subj_pickle
from load_trec import read_trec_pickle
from load_yelp import read_train_data, read_test_data
from load_tweet import read_tweet_pickle

ROTTEN_TOMATOES = 'rotten'
CUSTOMER_REVIEW = 'cr'
MPQA = 'mpqa'
SST_SENT = 'sst_sent'
SST_SENT_POL = 'sst_sent_polarity'
SST_KAGGLE = 'sst_kaggle'
TREC = 'trec'
SUBJ = 'subj'
IMDB = 'imdb'

all_datasets = [ROTTEN_TOMATOES, CUSTOMER_REVIEW, MPQA, SST_SENT, TREC, SUBJ]


def load_raw_datasets(datasets):
    if datasets == ROTTEN_TOMATOES:
        return read_rotten_pickle()
    elif datasets == CUSTOMER_REVIEW:
        return read_cr_pickle()
    elif datasets == MPQA:
        return read_mpqa_pickle()
    elif datasets == SST_SENT:
        return read_sst_sent_pickle()
    elif datasets == SST_SENT_POL:
        return read_sst_sent_pickle(polarity=True)
    elif datasets == TREC:
        return read_trec_pickle()
    elif datasets == SUBJ:
        return read_subj_pickle()
    elif datasets == IMDB:
        return read_imdb_pickle()


def count_length():
    import numpy
    for datasets in all_datasets:
        raw_data = load_raw_datasets(datasets)
        raw_train_sent = raw_data[0]
        sent_length = [len(x) for x in raw_train_sent]
        print datasets, "avg sentence length is", numpy.mean(sent_length)


if __name__ == '__main__':
    count_length()
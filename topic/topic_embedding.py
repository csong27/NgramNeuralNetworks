from corpus import MyCorpus
from train_lda import load_lda
from io_utils.load_data import *
from sklearn.feature_extraction import DictVectorizer
import cPickle as pkl

lda_pickled_path = "D:/data/nlpdata/pickled_data/topic/ldavec_"


def topic_vectorizer(data=SST_KAGGLE):
    lda = load_lda(data=data)
    train_doc, _, test_doc = read_sst_kaggle_pickle(use_textblob=True)
    docs = train_doc + test_doc
    corpus = MyCorpus(documents=docs)

    input_sent = []
    for doc in docs:
        bow_vec = corpus.dictionary.doc2bow(doc)
        dict_vec = dict(lda[bow_vec])
        input_sent.append(dict_vec)
    print "vectorizing topic probabilities..."
    dv = DictVectorizer()
    all_x = dv.fit_transform(input_sent)

    train_len = len(train_doc)
    train_x = all_x[:train_len]
    test_x = all_x[train_len:]

    return train_x.toarray(), test_x.toarray()


def topic_word_vectorizer(data=SST_KAGGLE):
    lda = load_lda(data=data)
    train_doc, _, test_doc = read_sst_kaggle_pickle(use_textblob=True)
    docs = train_doc + test_doc
    corpus = MyCorpus(documents=docs)
    count = 0
    input_sent = []
    for doc in docs:
        bow_vec = corpus.dictionary.doc2bow(doc)
        topics = lda[bow_vec]
        if len(topics) == 50:
            count += 1
    print count


def save_topic_vectors(data=SST_KAGGLE):
    train_x, test_x = topic_vectorizer(data=data)
    print train_x[:100]
    save_path = lda_pickled_path + data + ".pkl"
    f = open(save_path, "wb")
    pkl.dump((train_x, test_x), f, -1)
    f.close()


def read_topic_vectors(data=SST_KAGGLE):
    save_path = lda_pickled_path + data + ".pkl"
    print "reading topic vectors from %s" % save_path
    f = open(save_path, "rb")
    train_x, test_x = pkl.load(f)
    f.close()
    return train_x, test_x


if __name__ == '__main__':
    topic_word_vectorizer()
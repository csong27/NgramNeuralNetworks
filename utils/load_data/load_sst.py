#!/usr/bin/env python
# -*- coding: utf-8 -*-
from textblob import TextBlob
from sklearn.cross_validation import train_test_split
from path import Path
import os
import logging
from collections import namedtuple
import cPickle as pkl
from utils.preprocess import STOPWORDS


kaggle_train_path = Path('C:/Users/Song/Course/571/hw3/train.tsv')
kaggle_test_path = Path('C:/Users/Song/Course/571/hw3/test.tsv')
treebank_path = 'D:/data/nlpdata/stanfordSentimentTreebank/'
sst_sent_pickle = Path('C:/Users/Song/Course/571/project/pickled_data/sst_sent.pkl')
pickle_path = 'C:/Users/Song/Course/571/project/pickled_data/'


SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')


def construct_sentence_data(p):
    f = open(p)
    prev_sent_id = 0
    sentences = []
    labels = []
    f.readline()
    for line in f:
        data = line.split("\t")
        sent_id = data[1]
        text = data[2]
        label = data[-1]
        if sent_id != prev_sent_id:
            if int(sent_id) - int(prev_sent_id) > 1:
                print sent_id
            sentences.append(text)
            labels.append(int(label))
            prev_sent_id = sent_id
    print len(sentences)


def read_su_sentiment_rotten_tomatoes(dirname, lowercase=True):
    """
    Read and return documents from the Stanford Sentiment Treebank
    corpus (Rotten Tomatoes reviews), from http://nlp.Stanford.edu/sentiment/
    Initialize the corpus from a given directory, where
    http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
    has been expanded. It's not too big, so compose entirely into memory.
    """
    logging.info("loading corpus from %s" % dirname)

    # many mangled chars in sentences (datasetSentences.txt)
    chars_sst_mangled = ['à', 'á', 'â', 'ã', 'æ', 'ç', 'è', 'é', 'í',
                         'í', 'ï', 'ñ', 'ó', 'ô', 'ö', 'û', 'ü']
    sentence_fixups = [(char.encode('utf-8').decode('latin-1'), char) for char in chars_sst_mangled]

    # more junk, and the replace necessary for sentence-phrase consistency
    sentence_fixups.extend([
        ('Â', ''),
        ('\xa0', ' '),
        ('-LRB-', '('),
        ('-RRB-', ')'),
    ])
    # only this junk in phrases (dictionary.txt)
    phrase_fixups = [('\xa0', ' ')]

    # sentence_id and split are only positive for the full sentences

    # read sentences to temp {sentence -> (id,split) dict, to correlate with dictionary.txt
    info_by_sentence = {}
    with open(os.path.join(dirname, 'datasetSentences.txt'), 'r') as sentences:
        with open(os.path.join(dirname, 'datasetSplit.txt'), 'r') as splits:
            next(sentences)  # legend
            next(splits)     # legend
            for sentence_line, split_line in zip(sentences, splits):
                (id, text) = sentence_line.split('\t')
                id = int(id)
                text = text.rstrip()
                for junk, fix in sentence_fixups:
                    text = text.replace(junk, fix)
                (id2, split_i) = split_line.split(',')
                assert id == int(id2)
                if text not in info_by_sentence:    # discard duplicates
                    info_by_sentence[text] = (id, int(split_i))

    # read all phrase text
    phrases = [None] * 239232  # known size of phrases
    with open(os.path.join(dirname, 'dictionary.txt'), 'r') as phrase_lines:
        for line in phrase_lines:
            (text, id) = line.split('|')
            for junk, fix in phrase_fixups:
                text = text.replace(junk, fix)
            phrases[int(id)] = text.rstrip()  # for 1st pass just string

    SentimentPhrase = namedtuple('SentimentPhrase', SentimentDocument._fields + ('sentence_id',))
    # add sentiment labels, correlate with sentences
    with open(os.path.join(dirname, 'sentiment_labels.txt'), 'r') as sentiments:
        next(sentiments)  # legend
        for line in sentiments:
            (id, sentiment) = line.split('|')
            id = int(id)
            sentiment = float(sentiment)
            text = phrases[id]
            words = text.split()
            if lowercase:
                words = [word.lower() for word in words]
            (sentence_id, split_i) = info_by_sentence.get(text, (None, 0))
            split = [None, 'train', 'test', 'dev'][split_i]
            phrases[id] = SentimentPhrase(words, [id], split, sentiment, sentence_id)

    assert len([phrase for phrase in phrases if phrase.sentence_id is not None]) == len(info_by_sentence)  # all
    # counts don't match 8544, 2210, 1101 because 13 TRAIN and 1 DEV sentences are duplicates
    assert len([phrase for phrase in phrases if phrase.split == 'train']) == 8531  # 'train'
    assert len([phrase for phrase in phrases if phrase.split == 'test']) == 2210  # 'test'
    assert len([phrase for phrase in phrases if phrase.split == 'dev']) == 1100  # 'dev'

    logging.info("loaded corpus with %i sentences and %i phrases from %s"
                 % (len(info_by_sentence), len(phrases), dirname))
    return phrases


def get_int_label(score):
    if 0 <= score <= 0.2:
        return 0
    if 0.2 < score <= 0.4:
        return 1
    if 0.4 < score <= 0.6:
        return 2
    if 0.6 < score <= 0.8:
        return 3
    else:
        return 4


def save_sentence_pickle():
    phrases = read_su_sentiment_rotten_tomatoes(dirname=treebank_path)
    train_x = []
    train_y = []
    validate_x = []
    validate_y = []
    test_x = []
    test_y = []
    for phrase in phrases:
        label = get_int_label(phrase.sentiment)
        if phrase.split == 'train':
            train_x.append(phrase.words)
            train_y.append(label)
        elif phrase.split == 'dev':
            validate_x.append(phrase.words)
            validate_y.append(label)
        elif phrase.split == 'test':
            test_x.append(phrase.words)
            test_y.append(label)

    f = open('sst_sent.pkl', 'wb')
    pkl.dump((train_x, train_y), f, -1)
    pkl.dump((validate_x, validate_y), f, -1)
    pkl.dump((test_x, test_y), f, -1)
    f.close()


def read_sst_sent_pickle():
    f = open(sst_sent_pickle, 'rb')
    train_x, train_y = pkl.load(f)
    validate_x, validate_y = pkl.load(f)
    test_x, test_y = pkl.load(f)
    return train_x, train_y, validate_x, validate_y, test_x, test_y


def save_sst_kaggle_pickle(use_textblob=False):
    train_x, train_y = read_kaggle_train(use_textblob=use_textblob)
    test_x = read_kaggle_test(use_textblob=use_textblob)
    print len([x for x in train_x if x == []])
    print len([x for x in test_x if x == []])
    fname = "sst_kaggle.pkl"
    if use_textblob:
        fname = "lemma_" + fname
    save_path = pickle_path + fname
    f = open(Path(save_path), 'wb')
    pkl.dump((train_x, train_y), f, -1)
    pkl.dump((test_x, []), f, -1)
    f.close()


def read_sst_kaggle_pickle(use_textblob=False):
    fname = "sst_kaggle.pkl"
    if use_textblob:
        fname = "lemma_" + fname
    save_path = pickle_path + fname
    f = open(Path(save_path), 'rb')
    train_x, train_y = pkl.load(f)
    test_x, _ = pkl.load(f)
    return train_x, train_y, test_x


def textblob_preprocess(message):
    words = TextBlob(message.lower()).words
    # for each word, take its "base form" = lemma
    return [word.lemma for word in words if word not in STOPWORDS]


def read_kaggle_train(p=kaggle_train_path, use_textblob=False):
    f = open(p)
    f.readline()
    x = []
    y = []
    for line in f:
        data = line.split('\t')
        label = int(data[-1])
        review = data[2]
        if review[-1] == '\n':
            review = review[:-1]
        if use_textblob:
            review = textblob_preprocess(review)
        else:
            review = review.lower().split(' ')
        x.append(review)
        y.append(label)
    return x, y


def read_kaggle_test(p=kaggle_test_path, use_textblob=False):
    f = open(p)
    f.readline()
    x = []
    for line in f:
        data = line.split('\t')
        review = data[2]
        if review[-1] == '\n':
            review = review[:-1]
        if use_textblob:
            review = textblob_preprocess(review)
        else:
            review = review.lower().split(' ')
        x.append(review)
    return x


def read_kaggle_raw():
    # train dataset
    f = open(kaggle_train_path)
    f.readline()
    train_x = []
    train_y = []
    for line in f:
        data = line.split('\t')
        label = int(data[-1])
        review = data[2]
        if review[-1] == '\n':
            review = review[:-1]
        train_x.append(review)
        train_y.append(label)
    f.close()

    # test dataset
    f = open(kaggle_test_path)
    f.readline()
    test_x = []
    for line in f:
        data = line.split('\t')
        review = data[2]
        if review[-1] == '\n':
            review = review[:-1]
        test_x.append(review)

    return train_x, train_y, test_x

if __name__ == '__main__':
    save_sst_kaggle_pickle()
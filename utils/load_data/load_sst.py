#!/usr/bin/env python
# -*- coding: utf-8 -*-
from path import Path
import os
import logging
from collections import namedtuple
import cPickle as pkl

kaggle_train_path = Path('C:/Users/Song/Course/571/hw3/train.tsv')
kaggle_test_path = Path('C:/Users/Song/Course/571/hw3/test.tsv')
treebank_path = 'D:/data/nlpdata/stanfordSentimentTreebank/'
sst_sent_pickle = Path('C:/Users/Song/Course/571/project/pickled_data/sst_sent.pkl')


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

SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')


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
            split = [None,'train','test','dev'][split_i]
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

if __name__ == '__main__':
    read_sst_sent_pickle()
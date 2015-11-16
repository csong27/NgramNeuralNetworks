__author__ = 'Song'
import json
from path import Path
from gensim.parsing import *
from label_embedding.my_word2vec import STAR_LABELS
from utils import yelp_2013_test, yelp_2013_train
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import sent_tokenize

STOPWORDS = """
a about across after afterwards against all almost alone along already also although always am among amongst amoungst an and another any anyhow anyone anything anyway anywhere are around as at be
became because become becomes becoming been before beforehand behind being beside besides between beyond both bottom but by call can
cannot cant co computer could couldnt cry de describe
detail did didn do does doesn doing don done down due during
each eg eight either eleven else elsewhere empty enough etc even ever every everyone everything everywhere except few fifteen
fify fill find fire for former formerly forty found from front full further get give go
had has hasnt have he hence her here hereafter hereby herein hereupon hers herself him himself his how however hundred i ie
if in inc indeed interest into is it its itself keep last latter latterly ltd
just kg km made make many may me meanwhile might mill mine more moreover move much must my myself name namely
neither never nevertheless next nine no nobody none noone nor not nothing now nowhere of off
often on once only onto or other others otherwise our ours ourselves out over own part per
perhaps please put rather re quite rather really regarding
same say see seem seemed seeming seems several she should show side since sincere sixty so someone something sometime sometimes somewhere still such system
take ten that the their them themselves then thence there thereafter thereby therefore therein thereupon these they thick thin third this those though through throughout thru thus to together too toward towards twelve twenty un
until up upon us used using via
was we were what whatever when whence whenever where whereafter whereas whereby wherein whereupon wherever whether which while whither who whoever whole whom whose why will with within without would yet you
your yours yourself yourselves
"""
STOPWORDS = frozenset(w for w in STOPWORDS.split() if w)


def my_remove_stopwords(s):
    s = utils.to_unicode(s)
    return " ".join(w for w in s.split() if w not in STOPWORDS)


def my_strip_short(s, minsize=2):
    s = utils.to_unicode(s)
    return " ".join(e for e in s.split() if len(e) >= minsize)


DEFAULT_FILTERS = [lambda x: x.lower(), strip_punctuation, strip_multiple_whitespaces,
                   my_remove_stopwords, my_strip_short]

SIMPLE_FILTERS = [lambda x: x.lower(), strip_punctuation, strip_multiple_whitespaces]


def preprocess_review(text, filters=DEFAULT_FILTERS):
    return preprocess_string(text, filters=filters)


class MyDocuments(object):
    def __init__(self, filename, str_label=False, int_label=True):
        '''
        :param filename: input file
        :param str_label: yield iterator with last word as label
        :param int_label: yield tuple iterator, first element is label
        '''

        self.filename = Path(filename)
        if int_label and str_label:
            raise ValueError
        self.str_label = str_label
        self.int_label = int_label

    def __iter__(self):
        for line in open(self.filename):
            json_object = json.loads(line)
            text = json_object['text']
            star = json_object['stars']
            if self.int_label:
                arr = preprocess_review(text)
                yield (star, arr)
            elif self.str_label:
                arr = preprocess_review(text)
                arr.append(STAR_LABELS[int(star) - 1])
                yield arr
            else:
                review_id = json_object['review_id']
                yield TaggedDocument(words=preprocess_review(text), tags=[review_id])


class MySentences(object):
    def __init__(self, filename, document=True):
        '''
        :param filename: input file
        :param str_label: yield iterator with last word as label
        :param int_label: yield tuple iterator, first element is label
        '''
        self.filename = Path(filename)
        self.document = document

    def __iter__(self):
        for line in open(self.filename):
            json_object = json.loads(line)
            text = json_object['text']
            review_id = json_object['review_id']
            sentences = sent_tokenize(text)
            for i, sentence in enumerate(sentences):
                tag = ["%s;%d" % (review_id, i)]
                yield TaggedDocument(words=preprocess_review(sentence), tags=tag)


if __name__ == '__main__':
    sentences = MySentences(yelp_2013_train)
    for sentence in sentences:
        print sentence

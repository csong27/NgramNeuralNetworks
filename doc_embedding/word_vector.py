import cPickle as pkl
import numpy as np
from nltk.collocations import ngrams
from utils.load_data import *
from utils.load_vector_model import read_glove_model, read_google_model
from sklearn.cross_validation import train_test_split
from path import Path

max_count = 0


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


def get_document_matrix(text, model, cutoff=300, uniform=True, scale=0.1, shrink=True):
    matrix = None
    rand_vector = np.random.uniform(-scale, scale, model.vector_size) if uniform \
        else np.random.normal(0, scale, model.vector_size)
    count = 0
    if shrink and len(text) > cutoff:
        shrink_size = int(round(len(text) / float(cutoff) + 0.4))
        word_chunks = chunks(text, shrink_size)
        for chunk in word_chunks:
            avg_vector = np.zeros(model.vector_size)
            for word in chunk:
                word_vector = model[word] if word in model else rand_vector
                avg_vector += word_vector
            avg_vector /= len(chunk)
            if matrix is None:
                matrix = np.asarray([avg_vector])
            else:
                matrix = np.concatenate((matrix, [avg_vector]))
    else:
        for word in text:
            word_vector = model[word] if word in model else rand_vector
            if matrix is None:
                matrix = np.asarray([word_vector])
            else:
                matrix = np.concatenate((matrix, [word_vector]))
            count += 1
            if count >= cutoff:
                break

    if matrix is None:
        return np.zeros((cutoff, model.vector_size))
    length = matrix.shape[0]
    if length < cutoff:
        padding = np.zeros((cutoff - length, model.vector_size))
        matrix = np.concatenate((matrix, padding))
    elif length > cutoff:
        matrix = matrix[:cutoff]
    return matrix


def get_document_bigram_matrix(text, model, cutoff=50, uniform=True, scale=0.1):
    matrix = None
    rand_vector = np.random.uniform(-scale, scale, model.vector_size) if uniform \
        else np.random.normal(0, scale, model.vector_size)
    count = 0
    if len(text) == 1:
        text += ['!@#$']
    bigrams = ngrams(text, 2)
    for bigram in bigrams:
        word1, word2 = bigram
        word_vector_1 = model[word1] if word1 in model else rand_vector
        word_vector_2 = model[word2] if word2 in model else rand_vector
        bigram_vector = np.concatenate((word_vector_1, word_vector_2))
        if matrix is None:
            matrix = np.asarray([bigram_vector])
        else:
            matrix = np.concatenate((matrix, [bigram_vector]))
        count += 1
        if count >= cutoff:
            break
    if matrix is None:
        return np.zeros((cutoff, model.vector_size * 2))
    length = matrix.shape[0]
    if length < cutoff:
        padding = np.zeros((cutoff - length, model.vector_size * 2))
        matrix = np.concatenate((matrix, padding))
    elif length > cutoff:
        matrix = matrix[:cutoff]
    return matrix


def get_review_vector(text, model, average=True):
    count = 0
    vector = np.zeros(model.vector_size)
    for word in text:
        count += 1
        if word in model:
            vector += model[word]
    if average and len(text) > 0:
        vector /= len(text)
    global max_count
    if count > max_count:
        max_count = count
    return vector


def get_reviews_vectors(documents, model, average=True, aggregate=False, cutoff=300, uniform=True, bigram=False):
    for i in xrange(len(documents)):
        if aggregate:
            documents[i] = get_review_vector(documents[i], model, average)
        elif bigram:
            documents[i] = get_document_bigram_matrix(documents[i], model, cutoff=cutoff, uniform=uniform)
        else:
            documents[i] = get_document_matrix(documents[i], model, cutoff=cutoff, uniform=uniform)
    return documents


def get_aggregated_vectors(average=True, int_label=True, dim=300):
    model = read_glove_model(dim=dim)
    train_x, train_y, validate_x, validate_y = read_train_data(int_label=int_label)
    test_x, test_y = read_test_data(int_label=int_label)
    print "getting aggregate word vectors for documents..."
    train_x = get_reviews_vectors(train_x, model, average)
    validate_x = get_reviews_vectors(validate_x, model, average)
    test_x = get_reviews_vectors(test_x, model, average)
    return train_x, train_y, validate_x, validate_y, test_x, test_y


def get_document_matrices(google=False, dim=100, cutoff=60, uniform=True, data='rotten', cv=True, bigram=False, kaggle=False):
    print "getting concatenated word vectors for documents..."
    model = read_google_model() if google else read_glove_model(dim=dim)
    if cv:
        if data == 'rotten':
            x, y = read_rotten_pickle()
            cutoff = 56
        elif data == 'subj':
            x, y = read_subj_pickle()
        elif data == 'cr':
            x, y = read_cr_pickle()
        elif data == 'mpqa':
            x, y = read_mpqa_pickle()
            cutoff = 20
        else:
            raise NotImplementedError('Not such data set %s', data)
        x = get_reviews_vectors(x, model, aggregate=False, cutoff=cutoff, uniform=uniform, bigram=bigram)
        x = np.asarray(x)
        y = np.asarray(y)
        return x, y
    elif not kaggle:
        if data == 'imdb':
            train_x, train_y, validate_x, validate_y, test_x, test_y = read_imdb_pickle()
            cutoff = 75
        elif data == 'sst_sent':
            train_x, train_y, validate_x, validate_y, test_x, test_y = read_sst_sent_pickle()
        elif data == 'trec':
            train_x, train_y, validate_x, validate_y, test_x, test_y = read_trec_pickle()
            cutoff = 50
        else:
            raise NotImplementedError('Not such data set %s', data)
        train_x = get_reviews_vectors(train_x, model, aggregate=False, cutoff=cutoff, uniform=uniform, bigram=bigram)
        validate_x = get_reviews_vectors(validate_x, model, aggregate=False, cutoff=cutoff, uniform=uniform, bigram=bigram)
        test_x = get_reviews_vectors(test_x, model, aggregate=False, cutoff=cutoff, uniform=uniform, bigram=bigram)

        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y)
        validate_x = np.asarray(validate_x)
        validate_y = np.asarray(validate_y)
        test_x = np.asarray(test_x)
        test_y = np.asarray(test_y)

        return train_x, train_y, validate_x, validate_y, test_x, test_y
    elif kaggle:
        cutoff = 35
        train_x, train_y, test_x = read_sst_kaggle_pickle()
        train_x = get_reviews_vectors(train_x, model, aggregate=False, cutoff=cutoff, uniform=uniform, bigram=bigram)
        test_x = get_reviews_vectors(test_x, model, aggregate=False, cutoff=cutoff, uniform=uniform, bigram=bigram)
        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y)
        test_x = np.asarray(test_x)
        return train_x, train_y, test_x
    else:
        return Exception('Something went wrong')


def get_document_matrices_yelp(google=False, int_label=True, dim=50, cutoff=300, uniform=True, for_theano=True):
    model = read_google_model() if google else read_glove_model(dim=dim)
    train_x, train_y, validate_x, validate_y = read_train_data(int_label=int_label)
    test_x, test_y = read_test_data(int_label=int_label)
    print "getting concatenated word vectors for documents..."
    train_x = get_reviews_vectors(train_x, model, aggregate=False, cutoff=cutoff, uniform=uniform)
    validate_x = get_reviews_vectors(validate_x, model, aggregate=False, cutoff=cutoff, uniform=uniform)
    test_x = get_reviews_vectors(test_x, model, aggregate=False, cutoff=cutoff, uniform=uniform)

    if for_theano:
        train_y = np.asarray(train_y) - 1
        validate_y = np.asarray(validate_y) - 1
        test_y = np.asarray(test_y) - 1

    return train_x, train_y, validate_x, validate_y, test_x, test_y


def save_matrices_pickle(google=True, data='rotten', cv=True, bigram=False, kaggle=False):
    path = 'D:/data/nlpdata/pickled_data/'
    dataname = data + '_google.pkl' if google else data + '_glove.pkl'
    if bigram:
        dataname = 'bigram_' + dataname
    filename = Path(path + dataname)
    print 'saving data to %s...' % filename
    f = open(filename, 'wb')
    if cv:
        x, y = get_document_matrices(google=google, dim=300, data=data, bigram=bigram)
        print len(x)
        pkl.dump((x, y), f, -1)
    elif not kaggle:
        train_x, train_y, validate_x, validate_y, test_x, test_y = get_document_matrices(google=google, data=data,
                                                                                         cv=False, bigram=bigram)
        pkl.dump((train_x, train_y), f, -1)
        pkl.dump((validate_x, validate_y), f, -1)
        pkl.dump((test_x, test_y), f, -1)
    elif kaggle:
        train_x, train_y, test_x = get_document_matrices(google=google, data=data, cv=False,
                                                         bigram=bigram, kaggle=kaggle, dim=100)
        f_train = open('D:/data/nlpdata/pickled_data/kaggle/train_' + dataname, 'wb')
        f_test = open('D:/data/nlpdata/pickled_data/kaggle/test_' + dataname, 'wb')
        np.save(f_train, train_x)
        np.save(f_test, test_x)
        f_train.close()
        f_test.close()
    f.close()


def read_matrices_pickle(data='rotten', google=True, cv=True, bigram=False):
    path = 'D:/data/nlpdata/pickled_data/'
    filename = data + '_google.pkl' if google else data + '_glove.pkl'
    if bigram:
        filename = 'bigram_' + filename
    filename = Path(path + filename)
    print 'loading data from %s...' % filename
    f = open(filename, 'rb')
    if cv:
        x, y = pkl.load(f)
        return x, y
    else:
        train_x, train_y = pkl.load(f)
        validate_x, validate_y = pkl.load(f)
        test_x, test_y = pkl.load(f)
        return train_x, train_y, validate_x, validate_y, test_x, test_y


def read_matrices_kaggle_pickle():
    path = 'D:/data/nlpdata/pickled_data/kaggle/'
    train_filename = 'train_' + SST_KAGGLE + '_glove.pkl'
    filename = Path(path + train_filename)
    print 'loading data from %s...' % filename
    f = open(filename, 'rb')
    train_x = np.load(f)
    test_filename = 'test_' + SST_KAGGLE + '_glove.pkl'
    filename = Path(path + test_filename)
    print 'loading data from %s...' % filename
    f = open(filename, 'rb')
    test_x = np.load(f)
    _, train_y, _ = read_sst_kaggle_pickle()
    train_y = np.asarray(train_y)
    return train_x, train_y, test_x

if __name__ == '__main__':
    save_matrices_pickle(google=False, data=SST_KAGGLE, cv=False, kaggle=True)
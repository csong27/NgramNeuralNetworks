import cPickle as pkl
import numpy as np
from io_utils.load_data import *
from io_utils.load_vector_model import read_glove_model, read_google_model
from path import Path

import platform
if platform.system() == 'Windows':
    data_path = 'D:/data/nlpdata/pickled_data/'
else:
    data_path = '/home/scz8928999/data/pickled/matrices/'


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


def get_document_matrix(text, model, cutoff=300, uniform=False, scale=0.025, shrink=True):
    matrix = None
    rand_vector = np.random.uniform(-scale, scale, model.vector_size) if uniform \
        else np.random.normal(0, scale, model.vector_size)
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
        count = 0   # count word length
        for word in text:
            word_vector = model[word] if word in model else rand_vector
            if matrix is None:
                matrix = np.asarray([word_vector])
            else:
                matrix = np.concatenate((matrix, [word_vector]))
            count += 1
            if count >= cutoff:
                break
    # return random if matrix has no word
    if matrix is None:
        matrix = np.random.uniform(-scale, scale, (cutoff, model.vector_size)) if uniform\
            else np.random.normal(0, scale, (cutoff, model.vector_size))
        return matrix
    # add zero padding and cutoff
    length = matrix.shape[0]
    if length < cutoff:
        padding = np.zeros((cutoff - length, model.vector_size))
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


def get_reviews_vectors(documents, model, average=True, aggregate=False, cutoff=300, uniform=True):
    for i in xrange(len(documents)):
        if aggregate:
            documents[i] = get_review_vector(documents[i], model, average)
        else:
            documents[i] = get_document_matrix(documents[i], model, cutoff=cutoff, uniform=uniform)
    return documents


def get_aggregated_vectors(google=True, data=SST_KAGGLE, average=True, dim=300):
    if google:
        model = read_google_model()
    else:
        model = read_glove_model(dim=dim)
    print "getting aggregate word vectors for documents..."
    if data == SST_KAGGLE:
        train_x, train_y, test_x = read_sst_kaggle_pickle()
        train_x = get_reviews_vectors(train_x, model, average=average, aggregate=True)
        test_x = get_reviews_vectors(test_x, model, average=average, aggregate=True)
        return train_x, train_y, test_x


def save_aggregated_vectors(google=True, data=SST_KAGGLE, average=True, dim=300):
    path = 'D:/data/nlpdata/pickled_data/average_'
    path += 'google_' if google else 'glove_'
    path += data + '.pkl'
    if data == SST_KAGGLE:
        train_x, train_y, test_x = get_aggregated_vectors(google=google, data=data, average=average, dim=dim)
        f = open(path, 'wb')
        pkl.dump((train_x, train_y), f, -1)
        pkl.dump((test_x, []), f, -1)
        f.close()


def read_aggregated_vectors(google=True, data=SST_KAGGLE):
    path = 'D:/data/nlpdata/pickled_data/average_'
    path += 'google_' if google else 'glove_'
    path += data + '.pkl'
    f = open(path, 'rb')
    train_x, train_y = pkl.load(f)
    test_x, test_y = pkl.load(f)
    f.close()
    return train_x, train_y, test_x


def get_document_matrices(google=False, dim=100, cutoff=50, uniform=True, data='rotten', cv=True, huge=False):
    print "getting concatenated word vectors for documents..."
    model = read_google_model() if google else read_glove_model(dim=dim, huge=huge)
    if cv:
        if data == ROTTEN_TOMATOES:
            x, y = read_rotten_pickle()
            cutoff = 56
        elif data == SUBJ:
            x, y = read_subj_pickle()
        elif data == CUSTOMER_REVIEW:
            cutoff = 40
            x, y = read_cr_pickle()
        elif data == MPQA:
            x, y = read_mpqa_pickle()
            cutoff = 20
        else:
            raise NotImplementedError('Not such cross validation data set %s', data)
        x = get_reviews_vectors(x, model, aggregate=False, cutoff=cutoff, uniform=uniform)
        x = np.asarray(x)
        y = np.asarray(y)
        return x, y
    else:
        if data == IMDB:
            train_x, train_y, validate_x, validate_y, test_x, test_y = read_imdb_pickle()
            cutoff = 75
        elif data == SST_SENT:
            cutoff = 50
            train_x, train_y, validate_x, validate_y, test_x, test_y = read_sst_sent_pickle()
        elif data == SST_SENT_POL:
            cutoff = 50
            train_x, train_y, validate_x, validate_y, test_x, test_y = read_sst_sent_pickle(polarity=True)
        elif data == TREC:
            train_x, train_y, validate_x, validate_y, test_x, test_y = read_trec_pickle()
            cutoff = 30
        else:
            raise NotImplementedError('Not such train/dev/test data set %s', data)
        train_x = get_reviews_vectors(train_x, model, aggregate=False, cutoff=cutoff, uniform=uniform)
        validate_x = get_reviews_vectors(validate_x, model, aggregate=False, cutoff=cutoff, uniform=uniform)
        test_x = get_reviews_vectors(test_x, model, aggregate=False, cutoff=cutoff, uniform=uniform)

        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y)
        validate_x = np.asarray(validate_x)
        validate_y = np.asarray(validate_y)
        test_x = np.asarray(test_x)
        test_y = np.asarray(test_y)

        return train_x, train_y, validate_x, validate_y, test_x, test_y


def save_matrices_pickle(google=True, data='rotten', cv=True, dim=50, huge=False):
    path = data_path + str(dim) if dim != 300 else data_path
    if huge:
        path += "huge_"
    dataname = data + '_google.pkl' if google else data + '_glove.pkl'
    filename = Path(path + dataname)
    print 'saving data to %s...' % filename
    f = open(filename, 'wb')
    if cv:
        x, y = get_document_matrices(google=google, dim=dim, data=data, huge=huge)
        pkl.dump((x, y), f, -1)
    else:
        train_x, train_y, validate_x, validate_y, test_x, test_y = get_document_matrices(google=google, data=data,
                                                                                         cv=False, dim=dim, huge=huge)
        pkl.dump((train_x, train_y), f, -1)
        pkl.dump((validate_x, validate_y), f, -1)
        pkl.dump((test_x, test_y), f, -1)

    f.close()


def read_matrices_pickle(data='rotten', google=True, cv=True, dim=300, huge=False):
    path = data_path
    if dim != 300:
        path += str(dim)
    if huge:
        path += "huge_"
    filename = data + '_google.pkl' if google else data + '_glove.pkl'
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
    save_matrices_pickle(data=SST_SENT_POL, cv=False, google=False, dim=300, huge=True)

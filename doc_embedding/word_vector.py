import cPickle as pkl

import numpy as np
from utils.load_rotten import read_rotten_pickle

from utils.load_data.load_yelp import read_train_data, read_test_data
from utils.load_vector_model import read_glove_model, read_google_model

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


def get_reviews_vectors(documents, model, average=True, aggregate=True, cutoff=300, uniform=True):
    for i in xrange(len(documents)):
        if aggregate:
            documents[i] = get_review_vector(documents[i], model, average)
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


def get_document_matrices_rotten(google=False, dim=50, cutoff=50, uniform=True):
    model = read_google_model() if google else read_glove_model(dim=dim)
    x, y = read_rotten_pickle()
    print "getting concatenated word vectors for documents..."
    x = get_reviews_vectors(x, model, aggregate=False, cutoff=cutoff, uniform=uniform)
    x = np.asarray(x)
    y = np.asarray(y)
    return x, y


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


def save_matrices_pickle(google=True, data='rotten'):
    filename = data + '_google.pkl' if google else data + '_glove.pkl'
    if data == 'rotten':
        x, y = get_document_matrices_rotten(google=google, cutoff=56, dim=300)
        f = open(filename, 'wb')
        pkl.dump((x, y), f, -1)
    else:
        raise NotImplementedError


def read_matrices_pickle(data='rotten', google=True):
    filename = '../pickled_data/' + data + '_google.pkl' if google else '../pickled_data/' + data + '_glove.pkl'
    if data == 'rotten':
        print 'loading data from %s...' % filename
        f = open(filename, 'rb')
        x, y = pkl.load(f)
        return x, y
    else:
        raise NotImplementedError


if __name__ == '__main__':
    save_matrices_pickle(google=False)
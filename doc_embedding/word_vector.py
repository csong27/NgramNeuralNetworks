from utils.load_data import read_train_data, read_test_data
from utils.load_vector import read_glove_model
import numpy as np

max_count = 0


def get_document_matrix(text, model, cutoff=300, uniform=True, scale=0.25):
    matrix = None
    rand_vector = np.random.uniform(-scale, scale, model.vector_size) if uniform \
        else np.random.normal(0, scale, model.vector_size)
    count = 0
    for word in text:
        if word in model:
            if matrix is None:
                matrix = np.asarray([model[word]])
            else:
                matrix = np.concatenate((matrix, [model[word]]))
        else:
            if matrix is None:
                matrix = np.asarray([rand_vector])
            else:
                matrix = np.concatenate((matrix, [rand_vector]))
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


def get_document_matrices(int_label=True, dim=50, cutoff=300, uniform=True, for_theano=True):
    model = read_glove_model(dim=dim)
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


if __name__ == '__main__':
    # train_x, train_y, validate_x, validate_y, test_x, test_y = get_aggregated_vectors()
    # print max_count
    train_x, train_y, validate_x, validate_y, test_x, test_y = get_document_matrices()
    print test_x[0].shape
    print test_x[0]
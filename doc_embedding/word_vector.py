from utils.load_data import read_train_data, read_test_data
from utils.load_vector import read_glove_model
import numpy as np

max_count = 0


def get_document_matrix(text, model, cutoff=300, uniform=True):
    matrix = []
    zero_vector = np.zeros(model.vector_size)
    rand_vector = np.random.uniform(-0.25, 0.25, model.vector_size) if uniform \
        else np.random.normal(0, 0.25, model.vector_size)
    for word in text:
        if word in model:
            matrix.append(model[word])
        else:
            matrix.append(rand_vector)
    length = len(matrix)
    if length < cutoff:
        matrix += [zero_vector] * (cutoff - length)
    elif length > cutoff:
        matrix = matrix[:cutoff]
    return np.asarray(matrix)


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


def get_document_matrices(int_label=True, dim=300, cutoff=300, uniform=True):
    model = read_glove_model(dim=dim)
    train_x, train_y, validate_x, validate_y = read_train_data(int_label=int_label)
    test_x, test_y = read_test_data(int_label=int_label)
    print "getting concatenated word vectors for documents..."
    # train_x = get_reviews_vectors(train_x, model, aggregate=False, cutoff=cutoff, uniform=uniform)
    # validate_x = get_reviews_vectors(validate_x, model, aggregate=False, cutoff=cutoff, uniform=uniform)
    test_x = get_reviews_vectors(test_x, model, aggregate=False, cutoff=cutoff, uniform=uniform)
    return train_x, train_y, validate_x, validate_y, test_x, test_y


if __name__ == '__main__':
    # train_x, train_y, validate_x, validate_y, test_x, test_y = get_aggregated_vectors()
    # print max_count
    train_x, train_y, validate_x, validate_y, test_x, test_y = get_document_matrices()
    print test_x[0].shape
    print test_x[0]
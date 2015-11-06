from utils.loadData import read_train_data, read_test_data
from utils.loadVector import read_glove_model
import numpy as np


def get_review_vector(text, model, average=True):
    vector = np.zeros(model.vector_size)
    for word in text:
        if word in model:
            vector += model[word]
    if average:
        vector /= len(text)
    return vector


def get_reviews_vectors(documents, model, average=True):
    for i in xrange(len(documents)):
        documents[i] = get_review_vector(documents[i], model, average)
    return documents


def get_document_vectors(average=True):
    model = read_glove_model()
    train_x, train_y, validate_x, validate_y = read_train_data()
    test_x, test_y = read_test_data()
    print "getting aggregate word vectors for documents..."
    train_x = get_reviews_vectors(train_x, model, average)
    validate_x = get_reviews_vectors(validate_x, model, average)
    test_x = get_reviews_vectors(test_x, model, average)
    return train_x, train_y, validate_x, validate_y, test_x, test_y

if __name__ == '__main__':
    train_x, train_y, validate_x, validate_y, test_x, test_y = get_document_vectors()
    print train_x[0].shape
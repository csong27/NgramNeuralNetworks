from utils.loadData import read_train_data, read_test_data
from utils.loadVector import read_glove_model
from nltk.collocations import ngrams
import numpy as np


def get_bigram_review_vector(text, model, average=True, params=(1, 1)):
    bigrams = ngrams(text, 2)
    vector = np.zeros(model.vector_size)
    count = 0
    for bigram in bigrams:
        bigram_vector = np.zeros(model.vector_size)
        if bigram[0] in model:
            bigram_vector += model[bigram[0]] * params[0]
        if bigram[1] in model:
            bigram_vector += model[bigram[1]] * params[1]
        count += 1
        vector += bigram_vector
    if average and count > 0:
        vector /= count
    return vector


def get_reviews_vectors(documents, model, average=True, params=(1, 1)):
    for i in xrange(len(documents)):
        documents[i] = get_bigram_review_vector(documents[i], model, average, params=params)
    return documents


def get_convolutional_bigram_vectors(average=True, int_label=True, dim=300, params=(1, 1)):
    model = read_glove_model(dim=dim)
    train_x, train_y, validate_x, validate_y = read_train_data(int_label=int_label)
    test_x, test_y = read_test_data(int_label=int_label)
    print "getting bigram word vectors for documents..."
    train_x = get_reviews_vectors(train_x, model, average=average, params=params)
    validate_x = get_reviews_vectors(validate_x, model, average=average, params=params)
    test_x = get_reviews_vectors(test_x, model, average=average, params=params)
    return train_x, train_y, validate_x, validate_y, test_x, test_y

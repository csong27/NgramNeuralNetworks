from utils.load_data import read_train_data, read_test_data
from utils.load_vector import read_glove_model
import numpy as np

max_count = 0


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


def get_reviews_vectors(documents, model, average=True):
    for i in xrange(len(documents)):
        documents[i] = get_review_vector(documents[i], model, average)
    return documents


def get_aggregated_vectors(average=True, int_label=True, dim=300):
    model = read_glove_model(dim=dim)
    train_x, train_y, validate_x, validate_y = read_train_data(int_label=int_label)
    test_x, test_y = read_test_data(int_label=int_label)
    print "getting aggregate word vectors for documents..."
    # train_x = get_reviews_vectors(train_x, model, average)
    # validate_x = get_reviews_vectors(validate_x, model, average)
    test_x = get_reviews_vectors(test_x, model, average)
    return train_x, train_y, validate_x, validate_y, test_x, test_y

if __name__ == '__main__':
    train_x, train_y, validate_x, validate_y, test_x, test_y = get_aggregated_vectors()
    print max_count

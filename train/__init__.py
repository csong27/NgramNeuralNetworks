import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from io_utils.word_index import read_word2index_data
from io_utils.load_data import *


def main_loop(n_epochs, n_train_batches, train_model, val_model, test_model, set_zero, zero_vec, shuffle_batch=True,
              validation_only=False, predict_model=None):
    epoch = 0
    best_val_accuracy = 0
    test_accuracy = 0
    best_prediction = None
    while epoch < n_epochs:
        epoch += 1
        cost_list = []
        # shuffle mini-batch if specified
        indices = np.random.permutation(range(n_train_batches)) if shuffle_batch else xrange(n_train_batches)
        for minibatch_index in indices:
            cost_mini = train_model(minibatch_index)
            cost_list.append(cost_mini)
            set_zero(zero_vec)  # reset the zero vectors
        val_accuracy = 1 - val_model(epoch)
        # get a best result
        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            test_accuracy = 1 - test_model(epoch)
            if predict_model is not None:
                best_prediction = predict_model(epoch)
        cost_epoch = np.mean(cost_list)
        print 'epoch %i, train cost %f, validate accuracy %f' % (epoch, cost_epoch, val_accuracy * 100.)
    if predict_model is not None:
        return best_prediction
    # return values
    if validation_only:
        return best_val_accuracy
    print "\nbest test accuracy is %f" % test_accuracy
    return test_accuracy


def prepare_datasets(data, resplit=True, validation_ratio=0.2):
    datasets, W, mask = read_word2index_data(data=data, google=True, cv=False)
    train_x, train_y, validate_x, validate_y, test_x, test_y = datasets
    if data == TREC and resplit:
        train_x, train_y, validate_x, validate_y, mask = resplit_train_data(train_x, train_y, validate_x, validate_y,
                                                                            validation_ratio, mask=mask)
    return train_x, train_y, validate_x, validate_y, test_x, test_y, W, mask


def resplit_train_data(train_x, train_y, validate_x, validate_y, validate_ratio, mask=None):
    all_x = np.concatenate((train_x, validate_x), axis=0)
    all_y = np.concatenate((train_y, validate_y))
    sss_indices = StratifiedShuffleSplit(y=all_y, n_iter=1, test_size=validate_ratio)
    for indices in sss_indices:
        train_index, test_index = indices
    train_x = all_x[train_index]
    validate_x = all_x[test_index]
    train_y = all_y[train_index]
    validate_y = all_y[test_index]
    if mask is not None:
        train_mask, validate_mask, test_mask = mask
        all_mask = np.concatenate((train_mask, validate_mask), axis=0)
        train_mask = all_mask[train_index]
        validate_mask = all_mask[test_index]
        mask = (train_mask, validate_mask, test_mask)
    return train_x, train_y, validate_x, validate_y, mask


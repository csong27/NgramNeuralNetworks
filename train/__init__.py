import numpy as np


def main_loop(n_epochs, n_train_batches, train_model, val_model, test_model, set_zero, zero_vec, shuffle_batch=True,
              validation_only=False):
    epoch = 0
    best_val_accuracy = 0
    test_accuracy = 0
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
        cost_epoch = np.mean(cost_list)
        print 'epoch %i, train cost %f, validate accuracy %f' % (epoch, cost_epoch, val_accuracy * 100.)
    # return values
    if validation_only:
        return best_val_accuracy
    print "\nbest test accuracy is %f" % test_accuracy
    return test_accuracy

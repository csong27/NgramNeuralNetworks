from neural_network import *
from dropout_net import get_concatenated_document_vectors
from utils.load_data import *
from path import Path


def train_lecun_net(
        img_size,
        datasets,
        n_epochs=25,
        shuffle_batch=True,
        batch_size=50,
        n_hidden=100,
        n_out=2,
        activation=relu,
        nkerns=[10, 20],
        update_rule='adadelta',
        lr_rate=0.01,
        momentum_ratio=0.9,
        no_test_y=False
):
    rng = np.random.RandomState(23455)
    if no_test_y:
        train_x, train_y, validate_x, validate_y, test_x = datasets
        test_y = np.asarray([])
    else:
        train_x, train_y, validate_x, validate_y, test_x, test_y = datasets

    print 'size of train, validation, test set are %d, %d, %d' % (train_y.shape[0], validate_y.shape[0], test_x.shape[0])

    # if dataset size is not a multiple of mini batches, replicate
    residual = train_x.shape[0] % batch_size
    if residual > 0:
        extra_data_num = batch_size - residual
        extra_indices = np.random.permutation(np.arange(train_x.shape[0]))[:extra_data_num]
        extra_x = train_x[extra_indices]
        extra_y = train_y[extra_indices]
        train_x = np.append(train_x, extra_x, axis=0)
        train_y = np.append(train_y, extra_y, axis=0)

    train_x, train_y = shared_dataset((train_x, train_y))
    validate_x, validate_y = shared_dataset((validate_x, validate_y))
    test_x, test_y = shared_dataset((test_x, test_y))

    n_train_batches = train_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    print 'building network, number of mini-batches are %d...' % n_train_batches

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of

    layer0_input = x.reshape((batch_size, 1, img_size, img_size))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (30-7+1 , 30-7+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, img_size, img_size),
        filter_shape=(nkerns[0], 1, 7, 7),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 4 * 4,
        n_out=n_hidden,
        activation=activation
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=n_hidden, n_out=n_out)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    if update_rule == 'adadelta':
        grad_updates = adadelta(loss_or_grads=cost, params=params, learning_rate=lr_rate, rho=0.95, epsilon=1e-6)
    elif update_rule == 'adagrad':
        grad_updates = adagrad(loss_or_grads=cost, params=params, learning_rate=lr_rate, epsilon=1e-6)
    elif update_rule == 'adam':
        grad_updates = adam(loss_or_grads=cost, params=params, learning_rate=lr_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif update_rule == 'momentum':
        grad_updates = momentum(loss_or_grads=cost, params=params, momentum=momentum_ratio, learning_rate=lr_rate)
    elif update_rule == 'sgd':
        grad_updates = sgd(loss_or_grads=cost, params=params, learning_rate=lr_rate)
    else:
        raise NotImplementedError("This optimization method is not implemented %s" % update_rule)

    train_model = theano.function([index], cost, updates=grad_updates,
                                  givens={
                                      x: train_x[index * batch_size:(index + 1) * batch_size],
                                      y: train_y[index * batch_size:(index + 1) * batch_size]
                                  })

    val_model = theano.function([index], layer3.errors(y), on_unused_input='ignore', givens={x: validate_x, y: validate_y})

    test_model = theano.function([index], layer3.errors(y), on_unused_input='ignore', givens={x: test_x, y: test_y})

    predict_output = layer3.y_pred
    predict_model = theano.function([index], predict_output, on_unused_input='ignore', givens={x: test_x, y: test_y})

    print 'training with %s...' % update_rule
    epoch = 0
    best_val_accuracy = 0
    test_accuracy = 0
    best_prediction = None
    while epoch < n_epochs:
        epoch += 1
        cost_list = []
        indices = np.random.permutation(range(n_train_batches)) if shuffle_batch else xrange(n_train_batches)
        for minibatch_index in indices:
            cost_mini = train_model(minibatch_index)
            cost_list.append(cost_mini)
        val_accuracy = 1 - val_model(epoch)
        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            if not no_test_y:
                test_accuracy = 1 - test_model(epoch)
            else:
                best_prediction = predict_model(epoch)
        cost_epoch = np.mean(cost_list)
        print 'epoch %i, train cost %f, validate accuracy %f' % (epoch, cost_epoch, val_accuracy * 100.)
    if not no_test_y:
        print "\nbest test accuracy is %f" % test_accuracy
        return test_accuracy
    else:
        return best_prediction


if __name__ == '__main__':
    train_x, validate_x, test_x = get_concatenated_document_vectors(data=SST_KAGGLE)
    _, train_y, _, validate_y, _ = read_sst_kaggle_pickle()

    train_y = np.asarray(train_y)
    validate_y = np.asarray(validate_y)

    dim = train_x[0].shape[0]
    print "input dimension is %d" % dim

    img_size = int(np.sqrt(dim))

    n_out = len(np.unique(validate_y))
    datasets = (train_x, train_y, validate_x, validate_y, test_x)

    best_prediction = train_lecun_net(
        img_size=img_size,
        datasets=datasets,
        n_epochs=40,
        lr_rate=0.05,
        n_out=n_out,
        n_hidden=300,
        activation=leaky_relu,
        batch_size=100,
        update_rule='adagrad',
        no_test_y=True
    )
    import csv
    save_path = Path('C:/Users/Song/Course/571/hw3/kaggle_result.csv')
    with open(save_path, 'wb') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['PhraseId', 'Sentiment'])
        phrase_ids = np.arange(156061, 222353)
        for phrase_id, sentiment in zip(phrase_ids, best_prediction):
            writer.writerow([phrase_id, sentiment])

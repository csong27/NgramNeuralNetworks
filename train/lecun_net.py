from neural_network import *
from dropout_net import get_concatenated_document_vectors
from utils.load_data import *
from path import Path


def get_evaluate_model(tensor_variables, conv_layer, classifier, evaluate_set, predict=False, img_size=30):
    index = T.lscalar()
    x, y = tensor_variables
    evaluate_x, evaluate_y = evaluate_set
    test_size = evaluate_x.get_value(borrow=True).shape[0]
    test_layer_input = x.reshape((test_size, 1, img_size, img_size))
    test_layer_output = conv_layer.predict(test_layer_input, test_size, 1)
    test_y_pred = classifier.predict(test_layer_output.flatten(2))
    test_error = T.mean(T.neq(test_y_pred, y))
    test_model = theano.function([index], test_error, on_unused_input='ignore', givens={x: evaluate_x, y: evaluate_y})
    predict_model = theano.function([index], test_y_pred, on_unused_input='ignore', givens={x: evaluate_x, y: evaluate_y})
    if predict:
        return test_model, predict_model
    else:
        return test_model


def train_lecun_net(
        datasets,
        img_size,
        filter_size=13,
        pool_size=2,
        n_epochs=25,
        shuffle_batch=True,
        user_bias=False,
        batch_size=50,
        n_hidden=100,
        n_out=2,
        activation=relu,
        nkerns=10,
        dropout_rate=0.5,
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

    # if dataset size is not a multiple of mini batches, replicate
    residual = train_x.shape[0] % batch_size
    if residual > 0:
        extra_data_num = batch_size - residual
        extra_indices = np.random.permutation(np.arange(train_x.shape[0]))[:extra_data_num]
        extra_x = train_x[extra_indices]
        extra_y = train_y[extra_indices]
        train_x = np.append(train_x, extra_x, axis=0)
        train_y = np.append(train_y, extra_y, axis=0)

    print 'size of train, validation, test set are %d, %d, %d' % (train_y.shape[0], validate_y.shape[0], test_x.shape[0])

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

    conv_layer_input = x.reshape((batch_size, 1, img_size, img_size))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (30-13+1 , 30-13+1) = (18, 18)
    # maxpooling reduces this further to (18/2, 18/2) = (9, 9)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 9, 9)
    conv_layer = LeNetConvPoolLayer(
        rng,
        input=conv_layer_input,
        image_shape=(batch_size, 1, img_size, img_size),
        filter_shape=(nkerns, 1, filter_size, filter_size),
        poolsize=(pool_size, pool_size),
        activation=activation
    )
    assert (img_size - filter_size + 1) % pool_size == 0
    pool_out_size = (img_size - filter_size + 1) / pool_size

    mlp_input = conv_layer.output.flatten(2)
    n_in = nkerns * pool_out_size ** 2
    mlp = MLPDropout(
        rng=rng,
        input=mlp_input,
        layer_sizes=[n_in, n_hidden, n_out],
        dropout_rates=[dropout_rate],
        activations=[activation],
        use_bias=user_bias
    )
    # the cost we minimize during training is the NLL of the model
    cost = mlp.negative_log_likelihood(y)

    # create a list of all model parameters to be fit by gradient descent
    params = mlp.params + conv_layer.params

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

    val_model = get_evaluate_model(conv_layer=conv_layer, classifier=mlp, evaluate_set=(validate_x, validate_y),
                                   tensor_variables=(x, y), predict=False, img_size=img_size)
    test_model, predict_model = get_evaluate_model(conv_layer=conv_layer, classifier=mlp, evaluate_set=(test_x, test_y),
                                                   tensor_variables=(x, y), predict=True, img_size=img_size)

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


def wrapper_kaggle():
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
        filter_size=9,
        n_epochs=30,
        lr_rate=0.025,
        n_out=n_out,
        dropout_rate=0.5,
        n_hidden=300,
        nkerns=16,
        activation=leaky_relu,
        batch_size=100,
        update_rule='adagrad',
        user_bias=True,
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

if __name__ == '__main__':
    wrapper_kaggle()

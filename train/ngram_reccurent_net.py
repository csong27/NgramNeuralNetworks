from neural_network import *
from train import main_loop
import theano
import numpy as np


def train_ngram_rec_net(
        datasets,
        U,
        input_shape,
        ngrams=(2, 1),
        ngram_activation=tanh,
        non_static=True,
        n_epochs=25,
        n_kernels=(4, 3),
        ngram_out=(300, 200),
        mean_pool=False,
        shuffle_batch=True,
        batch_size=50,
        n_hidden=100,
        n_out=2,
        dropout_rate=0.3,
        update_rule='adadelta',
        rec_type='lstm',
        lr_rate=0.01,
        momentum_ratio=0.9,
        l2_ratio=1e-4,
        pool=True,
        validation_only=False,
        mask=None
):
    rng = np.random.RandomState(23455)
    train_x, train_y, validate_x, validate_y, test_x, test_y = datasets
    if mask is not None:
        print "using mask..."
        m = T.matrix('mask')
        train_mask, validate_mask, test_mask = mask
        train_mask = sharedX(train_mask)
        validate_mask = sharedX(validate_mask)
        test_mask = sharedX(test_mask)
    else:
        m = None

    print 'size of train, validation, test set are %d, %d, %d' % (train_y.shape[0], validate_y.shape[0], test_x.shape[0])

    train_x, train_y = shared_dataset((train_x, train_y))
    validate_x, validate_y = shared_dataset((validate_x, validate_y))
    test_x, test_y = shared_dataset((test_x, test_y))

    n_train_batches = train_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    print 'building network, number of mini-batches are %d...' % n_train_batches

    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')
    y = T.ivector('y')

    # word vectors
    Words = theano.shared(value=U.astype(theano.config.floatX), name="Words")
    # reset zero padding
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(U.shape[1]).astype(theano.config.floatX)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0, :], zero_vec_tensor))])

    ngram_input = Words[T.cast(x, dtype="int32")]

    # declare ngram recurrent network
    ngram_net = NgramRecurrentNetwork(
        rng=rng,
        input=ngram_input,
        input_shape=input_shape,
        ngrams=ngrams,
        ngram_out=ngram_out,
        n_kernels=n_kernels,
        dropout_rate=dropout_rate,
        n_hidden=n_hidden,
        n_out=n_out,
        mean=mean_pool,
        ngram_activation=ngram_activation,
        rec_type=rec_type,
        mask=m,
        pool=pool
    )
    errors = ngram_net.errors
    cost = ngram_net.negative_log_likelihood(y)
    params = ngram_net.params
    # learning word vectors as well
    if non_static:
        params += [Words]
    # L2 norm
    if l2_ratio > 0:
        cost = l2_regularization(l2_ratio, params, cost)

    grad_updates = get_grad_updates(update_rule=update_rule, cost=cost, params=params, lr_rate=lr_rate,
                                    momentum_ratio=momentum_ratio)
    if mask is not None:
        # functions for training
        train_model = theano.function([index], cost, updates=grad_updates,
                                      givens={
                                          x: train_x[index * batch_size:(index + 1) * batch_size],
                                          y: train_y[index * batch_size:(index + 1) * batch_size],
                                          m: train_mask[index * batch_size:(index + 1) * batch_size]
                                      })
        val_model = theano.function([index], errors(y), on_unused_input='ignore', givens={x: validate_x,
                                                                                          y: validate_y,
                                                                                          m: validate_mask})

        test_model = theano.function([index], errors(y), on_unused_input='ignore', givens={x: test_x,
                                                                                           y: test_y,
                                                                                           m: test_mask})
    else:
        train_model = theano.function([index], cost, updates=grad_updates,
                                      givens={
                                          x: train_x[index * batch_size:(index + 1) * batch_size],
                                          y: train_y[index * batch_size:(index + 1) * batch_size]
                                      })
        val_model = theano.function([index], errors(y), on_unused_input='ignore', givens={x: validate_x, y: validate_y})
        test_model = theano.function([index], errors(y), on_unused_input='ignore', givens={x: test_x, y: test_y})

    print 'training with %s...' % update_rule
    return main_loop(n_epochs=n_epochs, train_model=train_model, val_model=val_model, test_model=test_model,
                     set_zero=set_zero, zero_vec=zero_vec, n_train_batches=n_train_batches, shuffle_batch=shuffle_batch,
                     validation_only=validation_only)


def train_reversed_ngram_rec_net(
        datasets,
        U,
        input_shape,
        ngrams=(2, ),
        ngram_activation=tanh,
        non_static=True,
        n_epochs=25,
        n_kernels=(4, ),
        ngram_out=(300, ),
        mean_pool=False,
        shuffle_batch=True,
        batch_size=50,
        rec_hidden=300,
        n_out=2,
        dropout_rate=0.3,
        update_rule='adadelta',
        rec_type='lstm',
        rec_activation=tanh,
        mlp_activation=tanh,
        lr_rate=0.01,
        momentum_ratio=0.9,
        l2_ratio=1e-4,
        mlp=False,
        mlp_hidden=200,
        concat_out=False,
        validation_only=False,
        mask=None
):
    rng = np.random.RandomState(23455)
    train_x, train_y, validate_x, validate_y, test_x, test_y = datasets
    if mask is not None:
        print "using mask..."
        m = T.matrix('mask')
    else:
        m = None

    print 'size of train, validation, test set are %d, %d, %d' % (train_y.shape[0], validate_y.shape[0], test_x.shape[0])

    train_x, train_y = shared_dataset((train_x, train_y))
    validate_x, validate_y = shared_dataset((validate_x, validate_y))
    test_x, test_y = shared_dataset((test_x, test_y))

    n_train_batches = train_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    print 'building network, number of mini-batches are %d...' % n_train_batches

    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')
    y = T.ivector('y')

    # word vectors
    Words = theano.shared(value=U.astype(theano.config.floatX), name="Words")
    # reset zero padding
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(U.shape[1]).astype(theano.config.floatX)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0, :], zero_vec_tensor))])

    ngram_input = Words[T.cast(x, dtype="int32")]

    # declare ngram recurrent network
    ngram_net = ReversedNgramRecurrentNetwork(
        rng=rng,
        input=ngram_input,
        input_shape=input_shape,
        ngrams=ngrams,
        ngram_out=ngram_out,
        n_kernels=n_kernels,
        dropout_rate=dropout_rate,
        rec_activation=rec_activation,
        mlp_activation=mlp_activation,
        rec_hidden=rec_hidden,
        mlp_hidden=mlp_hidden,
        n_out=n_out,
        mean=mean_pool,
        ngram_activation=ngram_activation,
        rec_type=rec_type,
        concat_out=concat_out,
        mask=m,
        mlp=mlp
    )
    errors = ngram_net.errors
    cost = ngram_net.negative_log_likelihood(y)
    params = ngram_net.params
    # learning word vectors as well
    if non_static:
        params += [Words]
    # L2 norm
    if l2_ratio > 0:
        cost = l2_regularization(l2_ratio, params, cost)

    grad_updates = get_grad_updates(update_rule=update_rule, cost=cost, params=params, lr_rate=lr_rate,
                                    momentum_ratio=momentum_ratio)
    if mask is not None:
        train_mask, validate_mask, test_mask = mask
        train_mask = sharedX(train_mask)
        validate_mask = sharedX(validate_mask)
        test_mask = sharedX(test_mask)
        # functions for training
        train_model = theano.function([index], cost, updates=grad_updates,
                                      givens={
                                          x: train_x[index * batch_size:(index + 1) * batch_size],
                                          y: train_y[index * batch_size:(index + 1) * batch_size],
                                          m: train_mask[index * batch_size:(index + 1) * batch_size]
                                      })
        val_model = theano.function([index], errors(y), on_unused_input='ignore', givens={x: validate_x,
                                                                                          y: validate_y,
                                                                                          m: validate_mask})

        test_model = theano.function([index], errors(y), on_unused_input='ignore', givens={x: test_x,
                                                                                           y: test_y,
                                                                                           m: test_mask})
    else:
        train_model = theano.function([index], cost, updates=grad_updates,
                                      givens={
                                          x: train_x[index * batch_size:(index + 1) * batch_size],
                                          y: train_y[index * batch_size:(index + 1) * batch_size]
                                      })
        val_model = theano.function([index], errors(y), on_unused_input='ignore', givens={x: validate_x, y: validate_y})
        test_model = theano.function([index], errors(y), on_unused_input='ignore', givens={x: test_x, y: test_y})

    print 'training with %s...' % update_rule
    return main_loop(n_epochs=n_epochs, train_model=train_model, val_model=val_model, test_model=test_model,
                     set_zero=set_zero, zero_vec=zero_vec, n_train_batches=n_train_batches, shuffle_batch=shuffle_batch,
                     validation_only=validation_only)


from convolutional_net import train_ngram_conv_net
from neural_network.non_linear import *
from doc_embedding import *


def wrapper(data=TREC):
    train_x, train_y, validate_x, validate_y, test_x, test_y = read_matrices_pickle(google=False, data=data, cv=False)
    dim = train_x[0].shape[1]
    n_out = len(np.unique(test_y))
    shuffle_indices = np.random.permutation(train_x.shape[0])
    datasets = (train_x[shuffle_indices], train_y[shuffle_indices], validate_x, validate_y, test_x, test_y)
    train_ngram_conv_net(
        datasets=datasets,
        ngrams=(2, 1),
        use_bias=True,
        ngram_bias=False,
        dim=dim,
        lr_rate=0.01,
        dropout=True,
        dropout_rate=0.5,
        n_hidden=200,
        n_out=n_out,
        activation=tanh,
        ngram_activation=leaky_relu,
        batch_size=100,
        update_rule='adagrad'
    )


def wrapper_yelp():
    dim = 50
    cutoff = 50
    datasets = get_document_matrices_yelp(dim=dim, cutoff=cutoff)
    train_ngram_conv_net(
        datasets=datasets,
        ngrams=(2, 1),
        use_bias=False,
        ngram_bias=False,
        dim=dim,
        lr_rate=0.001,
        dropout=True,
        dropout_rate=0.5,
        n_hidden=200,
        activation=relu,
        ngram_activation=leaky_relu,
        batch_size=50,
        update_rule='adagrad'
    )


def wrapper_kaggle():
    train_x, train_y, validate_x, validate_y, test_x = read_matrices_kaggle_pickle()
    dim = train_x[0].shape[1]
    n_out = len(np.unique(validate_y))
    datasets = (train_x, train_y, validate_x, validate_y, test_x)

    best_prediction = train_ngram_conv_net(
        datasets=datasets,
        ngrams=(1, 2),
        use_bias=True,
        n_epochs=30,
        ngram_bias=False,
        dim=dim,
        lr_rate=0.05,
        n_out=n_out,
        dropout=True,
        dropout_rate=0.5,
        n_hidden=100,
        activation=leaky_relu,
        ngram_activation=leaky_relu,
        batch_size=50,
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

if __name__ == '__main__':
    wrapper_kaggle()
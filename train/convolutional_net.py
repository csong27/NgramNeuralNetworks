from neural_network import *
from doc_embedding import get_document_matrices


def train_bigram_conv_net(batch_size=50, dim=50):
    train_x, train_y, validate_x, validate_y, test_x, test_y = get_document_matrices(dim=dim)
    train_x, train_y = shared_dataset((train_x, train_y))
    validate_x, validate_y = shared_dataset((validate_x, validate_y))
    test_x, test_y = shared_dataset((test_x, test_y))

    n_train_batches = train_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    print n_train_batches

    x = T.tensor3('x')
    y = T.ivector('y')



if __name__ == '__main__':
    train_bigram_conv_net()
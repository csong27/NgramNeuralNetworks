from load_data import load_raw_datasets, ROTTEN_TOMATOES, CUSTOMER_REVIEW, SST_SENT_POL, SST_SENT, SUBJ, MPQA, TREC,\
    IMDB, all_datasets
from load_vector_model import read_glove_model, read_google_model
from path import Path
import numpy as np
import cPickle as pkl
import platform


if platform.system() == 'Windows':
    data_path = Path('D:/data/nlpdata/word2index/')
else:
    data_path = Path('/home/scz8928999/data/word2index/')

cutoff_map = {
    ROTTEN_TOMATOES: 56,
    SST_SENT_POL: 50,
    SST_SENT: 50,
    CUSTOMER_REVIEW: 45,
    MPQA: 20,
    TREC: 30,
    SUBJ: 50,
    IMDB: 75
}


def load_vocabulary(datasets):
    if len(datasets) == 2:
        x = datasets[0]
    elif len(datasets) == 4:
        x = datasets[0] + datasets[2]
    elif len(datasets) == 6:
        x = datasets[0] + datasets[2] + datasets[4]
    else:
        raise Exception("Something went wrong")
    vocab = set([word for sent in x for word in sent])
    return list(vocab)


def get_word_matrix(model, vocab, k=300, scale=0.25):
    word2index = {}
    W = np.zeros((len(vocab) + 1, k))
    i = 1   # index 0 should zero vector
    for word in vocab:
        if word in model:
            W[i] = model[word]
        else:
            W[i] = np.random.uniform(-scale, scale, k)
        word2index[word] = i
        i += 1
    return W, word2index


def get_index_data(word2index, x, cutoff):
    index_matrix = np.zeros((len(x), cutoff), dtype='int32')
    mask_matrix = np.zeros((len(x), cutoff), dtype='float32')
    for i, sent in enumerate(x):
        for j, word in enumerate(sent):
            if j < cutoff:
                index_matrix[i][j] = word2index[word]
                mask_matrix[i][j] = 1
    return index_matrix, mask_matrix


def save_index_data_pickle(data, datasets, W, word2index, cutoff, cv=False, google=False):
    file_name = data + "_word2index.pkl"
    if google:
        file_name = "google_" + file_name
    file_path = data_path + file_name
    f = open(file_path, 'wb')
    if cv:
        x, y = datasets
        x, mask = get_index_data(word2index, x, cutoff)
        y = np.asarray(y)
        pkl.dump((x, y), f, -1)
        pkl.dump((W, mask), f, -1)
    else:
        train_x, train_y, validate_x, validate_y, test_x, test_y = datasets
        train_x, train_mask = get_index_data(word2index, train_x, cutoff)
        validate_x, validate_mask = get_index_data(word2index, validate_x, cutoff)
        test_x, test_mask = get_index_data(word2index, test_x, cutoff)
        train_y = np.asarray(train_y)
        validate_y = np.asarray(validate_y)
        test_y = np.asarray(test_y)
        pkl.dump((train_x, train_y), f, -1)
        pkl.dump((validate_x, validate_y), f, -1)
        pkl.dump((test_x, test_y), f, -1)
        pkl.dump((W, [train_mask, validate_mask, test_mask]), f, -1)
    f.close()


def read_word2index_data(data, cv, google=False):
    file_name = data + "_word2index.pkl"
    if google:
        file_name = "google_" + file_name
    file_path = data_path + file_name
    f = open(file_path, 'rb')
    if cv:
        x, y = pkl.load(f)
        W, mask = pkl.load(f)
        datasets = (x, y)
    else:
        train_x, train_y = pkl.load(f)
        validate_x, validate_y = pkl.load(f)
        test_x, test_y = pkl.load(f)
        W, mask = pkl.load(f)
        datasets = (train_x, train_y, validate_x, validate_y, test_x, test_y)
    return datasets, W, mask


def save_index_data(data, google=False, huge=False):
    print "loading raw data.."
    datasets = load_raw_datasets(data)
    print "reading vocab.."
    vocab = load_vocabulary(datasets)
    model = read_google_model() if google else read_glove_model(dim=300, huge=huge)
    print "getting word embedding matrix"
    W, word2index = get_word_matrix(model, vocab)
    cutoff = cutoff_map[data]
    print "saving word2index..."
    if data in [ROTTEN_TOMATOES, MPQA, CUSTOMER_REVIEW, SUBJ]:
        save_index_data_pickle(data, datasets, W, word2index, cutoff, cv=True, google=google)
    elif data in [SST_SENT, SST_SENT_POL, TREC]:
        save_index_data_pickle(data, datasets, W, word2index, cutoff, google=google)


if __name__ == '__main__':
    save_index_data(SST_SENT_POL, google=True)
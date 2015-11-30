from utils.preprocess import preprocess_review
import cPickle as pkl

save_path = 'D:/data/nlpdata/pickled_data/tweet.pkl'
train_path = 'D:/data/nlpdata/tweet/train.csv'


def save_tweet_pickle():
    train_x, train_y = read_tweet_raw()
    f = open(save_path, 'wb')
    pkl.dump((train_x, train_y), f, -1)
    f.close()


def read_tweet_pickle():
    f = open(save_path, 'rb')
    train_x, train_y = pkl.load(f)
    f.close()
    return train_x, train_y


def read_tweet_raw(cutoff=10):
    # train dataset
    f = open(train_path)
    train_x = []
    train_y = []
    for line in f:
        data = line.split('","')
        label = int(data[0][1])
        review = data[-1]
        if review[-1] == '\n':
            review = review[:-1]
        try:
            all_words = preprocess_review(review)
        except UnicodeDecodeError:
            continue
        if len(all_words) < cutoff:  # filter
            continue
        train_x.append(" ".join(all_words))
        train_y.append(label)
    f.close()

    return train_x, train_y


if __name__ == '__main__':
    save_tweet_pickle()

__author__ = 'Song'
import json
from utils import yelp_2013_test, yelp_2013_train
from prepareData import preprocess_review
from matplotlib import pyplot as plt


def plot_histogram(percentages, title='length', bins=50):
    plt.title(title)
    plt.hist(percentages, bins=bins)
    plt.show()


def read_test_data(p=yelp_2013_test, preprocess=True, int_label=True):
    print "reading raw testing data..."
    f = open(p)
    test_x = []
    test_y = []
    for line in f.xreadlines():
        json_object = json.loads(line)
        text = json_object['text']
        if preprocess:
            text = preprocess_review(text)
        star = json_object['stars']
        if int_label:
            star = int(star)
        test_x.append(text)
        test_y.append(star)
    return test_x, test_y


def read_train_data(p=yelp_2013_train, validate_ratio=0.2, preprocess=True, int_label=True):
    print "reading raw training data..."
    f = open(p)
    documents = {}
    count = 0.0
    for line in f.xreadlines():
        count += 1
        json_object = json.loads(line)
        text = json_object['text']
        if preprocess:
            text = preprocess_review(text)
        star = json_object['stars']
        if int_label:
            star = int(star)
        if star in documents:
            documents[star].append(text)
        else:
            documents[star] = [text, ]
    sample_size = validate_ratio * count
    train_x = []
    train_y = []
    validate_x = []
    validate_y = []
    for star in documents:
        d = documents[star]
        star_sample_size = int(len(d) / count * sample_size)
        train_x.extend(d[star_sample_size:])
        validate_x.extend(d[:star_sample_size])
        train_y.extend([star] * (len(d) - star_sample_size))
        validate_y.extend([star] * star_sample_size)
    return train_x, train_y, validate_x, validate_y

if __name__ == '__main__':
    test_x, test_y = read_test_data()
    print test_x[0]

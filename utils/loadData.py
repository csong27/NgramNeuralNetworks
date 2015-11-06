__author__ = 'Song'
import json
from path import Path
from matplotlib import pyplot as plt

yelp_2013_train = Path('D:\data\yelpdata\yelp_training_set\yelp_training_set_review.json')
yelp_2013_test = Path('D:\data\yelpdata\yelp_test_set\yelp_test_set_review.json')

yelp_2014 = Path('D:/data/yelpdata/yelp_academic_review.json')


def plot_histogram(percentages, title='length', bins=50):
    plt.title(title)
    plt.hist(percentages, bins=bins)
    plt.show()


def read_test_data(p=yelp_2013_test):
    f = open(p)
    test_x = []
    test_y = []
    for line in f.xreadlines():
        json_object = json.loads(line)
        text = json_object['text']
        star = json_object['stars']
        test_x.append(text)
        test_y.append(star)
    return test_x, test_y


def read_train_data(p=yelp_2013_train, validate_ratio=0.2):
    f = open(p)
    documents = {}
    count = 0.0
    for line in f.xreadlines():
        count += 1
        json_object = json.loads(line)
        text = json_object['text']
        star = json_object['stars']
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

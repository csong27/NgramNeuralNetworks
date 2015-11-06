__author__ = 'Song'
import json
from utils import yelp_2013_train, yelp_2013_test


class User(object):
    def __init__(self, user_id):
        self.user_id = user_id
        self.scores = []

    def add_score(self, score):
        self.scores.append(score)

    def __repr__(self):
        return self.user_id

    def __str__(self):
        return self.user_id

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.user_id == other.user_id

if __name__ == '__main__':
    user_set = set()
    f1 = open(yelp_2013_train)
    for line in f1.xreadlines():
        json_object = json.loads(line)
        user_id = json_object['user_id']
        user_set.add(User(user_id=user_id))
    f1.close()
    f2 = open(yelp_2013_test)
    for line in f2.xreadlines():
        json_object = json.loads(line)
        user = User(user_id=json_object['user_id'])
        if user in user_set:
            print user
        else:
            print "No"

    f2.close()
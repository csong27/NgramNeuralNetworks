import numpy as np
from scipy.sparse import *
from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer()

counts = [[3, 0, 1],
          [2, 0, 0],
          [3, 0, 0],
          [4, 0, 0],
          [3, 2, 0],
          [3, 0, 2]]


tfidf = transformer.fit_transform(counts)

concat = np.zeros((6, 3))


print hstack((tfidf, concat)).toarray()
import csv
import numpy as np
from path import Path


def save_csv(best_prediction, fname=''):
    save_path = Path('C:/Users/Song/Course/571/hw3/kaggle_result_' + fname + '.csv')
    with open(save_path, 'wb') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['PhraseId', 'Sentiment'])
        phrase_ids = np.arange(156061, 222353)
        for phrase_id, sentiment in zip(phrase_ids, best_prediction):
            writer.writerow([phrase_id, sentiment])

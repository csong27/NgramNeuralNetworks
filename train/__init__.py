__author__ = 'Song'
from doc_embedding import get_aggregated_vectors, get_naive_bigram_vectors


def get_data(vector="averaged_word", dim=300, kernel=(1, 1), average=True):
    if vector == "averaged_word":
        return get_aggregated_vectors(dim=dim, average=True)
    elif vector == "summed_word":
        return get_aggregated_vectors(dim=dim, average=False)
    elif vector == "naive_bigram":
        return get_naive_bigram_vectors(dim=dim, average=average, kernel=kernel)
    elif vector == "doc":
        raise NotImplementedError
    else:
        raise NotImplementedError

__author__ = 'Song'
from doc_embedding.aggregate_word import get_aggregated_vectors


def get_data(vector="averaged_word", dim=300):
    if vector == "averaged_word":
        return get_aggregated_vectors(dim=dim, average=True)
    elif vector == "summed_word":
        return get_aggregated_vectors(dim=dim, average=False)
    elif vector == "doc":
        raise NotImplementedError
    else:
        raise NotImplementedError
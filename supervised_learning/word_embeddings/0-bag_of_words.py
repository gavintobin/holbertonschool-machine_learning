#!/usr/bin/env python3
'''task 1'''

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    '''bag of words'''
    # make count vector
    vectorizer = CountVectorizer(vocabulary=vocab)

    # sentences to matrix
    embeddings = vectorizer.fit_transform(sentences).toarray()

    # get features
    features = vectorizer.get_feature_names_out()

    return embeddings, features

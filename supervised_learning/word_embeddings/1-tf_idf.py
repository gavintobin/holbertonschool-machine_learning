#!/usr/bin/env python3
'''task 2'''


from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def tf_idf(sentences, vocab=None):
    '''tf idf'''
    #  makeTF-IDF vect
    vectorizer = TfidfVectorizer(vocabulary=vocab, token_pattern=r'\b\w+\b')

    #  put them into embeddings
    embeddings = vectorizer.fit_transform(sentences).toarray()

    if vocab is None:
        # get features
        features = set(vectorizer.get_feature_names())
    else:
        features = vocab

    return embeddings, features
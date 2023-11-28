#!/usr/bin/env python3
'''task 2'''


from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def tf_idf(sentences, vocab=None):
    '''tf idf'''

    if vocab is None:
        vocab = []

        # make vocab
        for sentence in sentences:
            words = sentence.split()
            vocab.extend(words)

    #  makeTF-IDF vect
    vectorizer = TfidfVectorizer(vocabulary=vocab)

    #  put them into embeddings
    embeddings = vectorizer.fit_transform(sentences).toarray()

    # get features
    features = vectorizer.get_feature_names_out()

    return embeddings, features

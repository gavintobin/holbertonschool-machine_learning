#!/usr/bin/env python3
'''task 4'''
from gensim.models import FastText
from gensim.test.utils import common_texts
from keras.layers import Embedding
import numpy as np


def gensim_to_keras(model):
    '''keras model'''
    vocab_size, emb_dim = model.wv.vectors.shape

    embedding_matrix = np.zeros((vocab_size, emb_dim))
    for i, word in enumerate(model.wv.index_to_key):
        embedding_matrix[i] = model.wv[word]

    keras_embedding = Embedding(
        input_dim=vocab_size,
        output_dim=emb_dim,
        weights=[embedding_matrix],
        trainable=True
    )

    return keras_embedding

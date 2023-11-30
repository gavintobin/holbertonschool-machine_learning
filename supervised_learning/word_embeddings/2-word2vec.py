#!/usr/bin/env ptyhon3
'''task 3'''
from gensim.test.utils import common_texts
from gensim.models import Word2Vec


def word2vec_model(sentences,
                   size=100,
                   min_count=5,
                   window=5,
                   negative=5,
                   cbow=True,
                   iterations=5,
                   seed=0,
                   workers=1):
    '''word 2 vec'''
    model = Word2Vec(sentences=sentences,
                     vector_size=size,
                     min_count=min_count,
                     window=window,
                     negative=negative,
                     epochs=iterations,
                     seed=seed,
                     workers=workers)

    model.save('word2vec.model')

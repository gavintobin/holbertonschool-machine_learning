#!/usr/bin/env python3
'''task 4'''

from gensim.models import FastText
from gensim.test.utils import common_texts


def fasttext_model(sentences,
                   size=100,
                   min_count=5,
                   negative=5,
                   window=5,
                   cbow=True,
                   iterations=5,
                   seed=0,
                   workers=1):
    '''fast text boiii'''
    sg = 0 if cbow else 1
    model = FastText(sentences=sentences,
                     vector_size=size,
                     window=window,
                     min_count=min_count,
                     negative=negative,
                     epochs=iterations,
                     seed=seed,
                     workers=workers)
    return model

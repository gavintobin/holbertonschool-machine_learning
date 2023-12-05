#!/usr/bin/bleuenv python3
'''task2'''

from collections import Counter
import math
import numpy as np


def ngram_bleu(references, sentence, n):
    '''ngram blue'''
    cngrams = Counter(zip(*[sentence[i:] for i in range(n)]))
    ngrams = Counter()

    for ref in references:
        reference_ngrams = Counter(zip(*[ref[i:] for i in range(n)]))
        ngrams += reference_ngrams

    clipped = {ngram: min(cngrams[ngram],
                     ngrams[ngram]) for ngram in cngrams}

    precision = sum(clipped.values()) / max(1, sum(cngrams.values()))

    pf = min(len(ref) for ref in references)

    brev = np.exp(1 - (pf / len(sentence))) if len(sentence) < pf else 1

    bs = brev * precision

    return bs

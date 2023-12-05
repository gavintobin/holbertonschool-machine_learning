#!/usr/bin/env python3
'''task2'''

from collections import Counter
import math

def ngram_bleu(references, sentence, n):
    # Calculate precision for each n-gram
    precisions = []
    for i in range(1, n + 1):
        reference_ngrams = Counter()
        sentence_ngrams = Counter()

        # Count n-grams in references
        for reference in references:
            reference_ngrams.update(zip(*[reference[j:] for j in range(i)]))

        # Count n-grams in the proposed sentence
        sentence_ngrams.update(zip(*[sentence[j:] for j in range(i)]))

        # Calculate intersection between reference and sentence n-grams
        common_ngrams = sum((reference_ngrams & sentence_ngrams).values())

        # Calculate precision for the current n-gram order
        precision = common_ngrams / max(1, sum(sentence_ngrams.values()))
        precisions.append(precision)

    # Calculate the geometric mean of precisions
    geometric_mean = math.exp(sum(map(math.log, precisions)) / n)

    # Calculate brevity penalty
    closest_reference_length = min(references, key=lambda ref: abs(len(ref) - len(sentence)))
    brevity_penalty = min(1, len(sentence) / len(closest_reference_length))

    # Calculate BLEU score
    bleu_score = brevity_penalty * geometric_mean

    return bleu_score

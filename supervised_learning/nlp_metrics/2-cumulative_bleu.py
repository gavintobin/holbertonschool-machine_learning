#!/usr/bin/env python3
'''task 3'''

from collections import Counter
import numpy as np

def cumulative_bleu(references, sentence, n):
    bleu_scores = []

    for i in range(1, n + 1):
        candidate_ngrams = Counter(zip(*[sentence[j:] for j in range(i)]))
        reference_ngrams = Counter()

        for ref in references:
            reference_ngrams.update(zip(*[ref[j:] for j in range(i)]))

        common_ngrams = sum((candidate_ngrams & reference_ngrams).values())
        total_ngrams = sum(candidate_ngrams.values())

        precision = common_ngrams / max(1, total_ngrams)
        bleu_scores.append(precision)

    geometric_mean = np.exp(np.mean(np.log(bleu_scores)))

    # Calculate brevity penalty
    closest_reference_length = min(references, key=lambda ref: abs(len(ref) - len(sentence)))
    brevity_penalty = min(1, len(sentence) / len(closest_reference_length))

    # Calculate cumulative BLEU score
    cumulative_bleu_score = brevity_penalty * geometric_mean

    return cumulative_bleu_score

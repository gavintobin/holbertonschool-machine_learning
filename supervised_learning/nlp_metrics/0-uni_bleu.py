#!/usr/bin/env python3
'''task 1'''
from collections import Counter


def uni_bleu(references, sentence):
    '''calcs bleu score for sentence'''
    precision_sum = 0.0
    total_reference_length = 0
    sentence_length = len(sentence)

    # Calculate precision for each reference
    for reference in references:
        reference_ngrams = set(reference)
        total_reference_length += len(reference)

        # Calculate common unigrams
        common_ngrams = sum(1 for s in set(sentence) if s in reference_ngrams)

        precision = common_ngrams / len(sentence)
        precision_sum += precision

    # Calculate brevity penalty
    brevity_penalty = min(1, len(sentence) / total_reference_length)
    bleu_score = brevity_penalty * (precision_sum / len(references))

    return bleu_score

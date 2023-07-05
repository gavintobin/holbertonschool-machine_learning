#!/usr/bin/env python3
'''no yimmmportos'''


def early_stopping(cost, opt_cost, threshold, patience, count):
    '''early srop reg'''
    if cost > opt_cost - threshold:
        count += 1
    else:
        count = 0

    stop_early = False
    if count >= patience:
        stop_early = True

    return stop_early, count

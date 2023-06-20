#!/usr/bin/env python3
'''task 4'''
import numpy as np

def moving_average(data, beta):
    '''move the ave'''
    totave = []
    av = 0

    for i in range(len(data)):
        av = (beta * av) + ((1 - beta) * data[i])
        newav = av / (1 - (beta ** (i + 1)))
        totave.append(newav)

    return totave

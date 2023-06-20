#!/usr/bin/env python3
'''task 4'''
import numpy as np

def moving_average(data, beta):
    '''move the ave'''
    totave = []
    av = 0
    nb = 1

    for i in range(len(data)):
        av = beta * av + (1 - beta) * data[i]
        nb *= 1 - beta
        newav = av / nb
        totave.append(newav)

    return totave

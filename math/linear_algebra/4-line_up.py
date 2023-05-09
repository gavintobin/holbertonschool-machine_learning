#!/usr/bin/env python3
'''task fourrrrrrrrrrr;'''


def add_arrays(arr1, arr2):
    if len(arr1) == len(arr2):
        sum = []
        for i in range(len(arr1)):
            sum.append(arr1[i] + arr2[i])
        return sum
    else:
        return None


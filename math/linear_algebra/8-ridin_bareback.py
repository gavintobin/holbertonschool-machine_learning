#!/usr/bin/env python3
'''' bareback is how i ride'''


def mat_mul(mat1, mat2):
    '''mulmatmulnatmulmat'''
    for i in range(len(mat1)):
        '''row it'''
        for j in range(len(mat2[0])):
            '''col it'''
            for k in range(len(mat2)):
                '''it row 2nd mat'''
                res[i][j] = mat1[i][k] * mat2[k][j]
                return res
            
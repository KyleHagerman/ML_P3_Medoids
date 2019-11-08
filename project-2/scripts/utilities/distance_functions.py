#!/usr/bin/env python3


# IMPORTS


import sys
sys.path.append('../data_api')
sys.path.append('../../../data')

from data_api import DataApi
import math
import random
import pandas as pd


# CLASS

'''
    Calculate distances between points using various distance functions.
'''


class DistanceFunctions:


    def __init__(self):
        self.DEBUG = False
        self.data_api_impl = DataApi('../../data/')


    '''
    INPUT: two data rows to compare, row_1 and row_2

    OUTPUT: Manhattan distance between data points
    '''
    def get_manhattan_distance(self, row_1, row_2):
        pass


    '''
    INPUT: two data rows to compare, row_1 and row_2

    OUTPUT: Euclidean distance between data points
    '''
    def get_euclidean_distance(self, row_1, row_2):
        sum_diff_squareds = 0
        for index in range(len(row_1)):
            if not isinstance(row_1[index], str):
                sum_diff_squareds = sum_diff_squareds + math.pow(row_1[index] - row_2[index], 2)

        return round(math.sqrt(sum_diff_squareds), 10) # round to 10 digits after the decimal
        


# EXECUTE SCRIPT


if __name__ == '__main__':

    print('running distance functions...')
    distance_functions_impl = DistanceFunctions()

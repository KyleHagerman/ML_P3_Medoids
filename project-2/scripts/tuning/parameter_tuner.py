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
    This class handles tuning all algorithms.
'''


class ParameterTuner:


    def __init__(self):
    	self.DEBUG = False


    '''
    get parameters for data set

    INPUT:
        - data_set_name: name of data set to get algorithm parameters for
        - algorithm: name of algorithm to get parameters for

    OUTPUT:
        - list of parameter values for data set and algorithm combination
    '''
    def get_params(self, data_set_name, algorithm):
        if data_set_name == 'abalone':
            return [1, 5, 10, 25, 50]
        elif data_set_name == 'car':
            return [1, 5, 10, 20]
        elif data_set_name == 'forestfires':
            return [1, 5, 10, 20]
        elif data_set_name == 'machine':
            return [1, 5, 10, 20]
        elif data_set_name == 'segmentation':
            return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        elif data_set_name == 'wine':
            return [1, 5, 10, 25, 50, 100]

        raise Exception('ERROR: unknown data_set_name --> ' + str(data_set_name))


    '''
    return parameter key for algorithm

    INPUT:
        - algorithm: name of algorithm

    OUTPUT:
        - parameter key, i.e. 'k' for knn
        - will we have different parameters other than k?
    '''
    def get_parameter_key(self, algorithm):
        if algorithm == 'knn':
            return 'k'
        elif algorithm == 'enn':
            return 'k'
        elif algorithm == 'cnn':
            return 'k'
        elif algorithm == 'kmeans_knn':
            return 'k'
        elif algorithm == 'kmedoids_knn':
            return 'k'

        return 'PARAM'



# EXECUTE SCRIPT


if __name__ == '__main__':

    print('running parameter tuner...')
    parameter_tuner_impl = ParameterTuner()

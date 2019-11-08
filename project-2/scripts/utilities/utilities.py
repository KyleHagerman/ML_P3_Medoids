#!/usr/bin/env python3


# IMPORTS


import sys
sys.path.append('../data_api')
sys.path.append('../../../data')

from data_api import DataApi

import math
import pandas as pd
from random import shuffle
from operator import itemgetter
from statistics import mode, StatisticsError


# CLASS

'''
    Calculate distances between points using various distance functions.
'''


class Utilities:

    def __init__(self):
    	self.DEBUG = False
    	self.data_api_impl = DataApi('../../data/')


    # return the highest frequency class value from a list of values
    def get_mode(self, vals):
        mode_val = None
        try:
            # call statistics library method to get mode
            mode_val = mode(vals)
        except StatisticsError:
            # no unique mode, must return one of the modes by randomly selecting
            # get list of tuples where each tuple is (nn_labels_value, frequency)
            item_counts = [(i, vals.count(i)) for i in set(vals)]
            # sort the list of tuples by frequency in descending order
            sorted_item_counts = sorted(item_counts, key=itemgetter(1), reverse=True)
            # randomly select one of the highest frequency items (modes) and return
            return self.get_random_mode(sorted_item_counts)

        # the mode() library method worked so return that value
        return mode_val


    # return randomly selected mode given a sorted list of item counts
    def get_random_mode(self, sorted_item_counts):
        # get frequency of highest frequency value
        highest_freq_val = sorted_item_counts[0][1]
        mode_tuples = []

        for mode_tuple in sorted_item_counts:
            # if the items frequency is not the highest we can stop iterating
            if mode_tuple[1] != highest_freq_val:
                break
            # if tuple item is a mode then add it to the list of modes
            if mode_tuple[1] == highest_freq_val:
                mode_tuples.append(mode_tuple)

        # randomly shuffle the modes so we can return a random one
        shuffle(mode_tuples)
        # return mode value from randomly shuffled list
        return mode_tuples[0][0]


	# helper method - return true if val is number, false otherwise
    def is_number(self, val):
	    try:
	        float(val)
	        return True
	    except ValueError:
	        return False


    '''
    get min and max values for each column

    INPUT:
        - data_frame: data

    OUTPUT:
        - list of tuples, where each tuple is (min, max) for min/max values of each column
    '''
    def get_column_bounds(self, data_frame):
        col_min_maxes = [] # list of tuples of min/max values for each column
        for column_label, _ in data_frame.items():
            column_data = data_frame.loc[:, column_label].values
            col_min_maxes.append(self.get_min_max(column_data))
        # return list of tuples containing min/max values for each column in dataframe
        return col_min_maxes


    # get min/max values for specific column
    def get_min_max(self, column_data):
        max_val = 0
        min_val = sys.maxsize # really big number
        for val in column_data:
            if self.is_number(val):
                num_val = float(val)
                if num_val < min_val:
                    min_val = num_val
                if num_val > max_val:
                    max_val = num_val
        # return (min, max) tuple for column
        return (min_val, max_val)



# EXECUTE SCRIPT


if __name__ == '__main__':

    print('running utilities...')
    utilities_impl = Utilities()

    print(utilities_impl.is_number(1))
    print(utilities_impl.is_number('wine'))
    print(utilities_impl.is_number('4.5'))


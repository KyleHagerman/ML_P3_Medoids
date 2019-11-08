#!/usr/bin/env python3


# IMPORTS


import sys
sys.path.append('../data_api')
sys.path.append('../algorithms')
sys.path.append('../../../data')
sys.path.append('../preprocessing')
sys.path.append('../cross_validator')
sys.path.append('../../../data')

from data_api import DataApi
from k_nearest_neighbor import KNN
from cross_validator import CrossValidator
from preprocessor import Preprocessor

import math
import random
import pandas as pd


# CLASS

'''
    This class handles all things condensed knn.
'''


class CondensedKNN(KNN):


    def __init__(self):
    	KNN.__init__(self)
    	self.DEBUG = True
    	self.data_api_impl = DataApi('../../data/')


#	def get_condensed_training_set(self, training_set):
	#	distance_matrix = self.get_distance_matrix(training_set)
	#	return distance_matrix



    # run condensed knn using train_set and test_set
	#def get_CNN_set(self, train_set):
	#	return distance_matrix

    def get_condensed_training_set(self, raw_train_set):
	    our_list=[]
	    #print('raw training set shape = ' + str(raw_train_set.shape))
	    #print(raw_train_set)
	    #distance = self.get_distance_matrix(raw_train_set)
	    distance_matrix = self.get_distance_matrix(raw_train_set.loc[:, raw_train_set.columns != 'CLASS'])
	    for row_number in range(1, distance_matrix.shape[0]):
		    our_column = None
		    for column in range(1, distance_matrix.shape[0]): #should this be zero?
			    try:
			        if (our_column is None) and (raw_train_set.loc[row_number, 'CLASS'] != raw_train_set.loc[column, 'CLASS']):
				        ref = distance_matrix[row_number][column]
				        our_column = column
			        elif (raw_train_set.loc[row_number, 'CLASS'] != raw_train_set.loc[column, 'CLASS']):
				        if ref > distance_matrix[row_number][column]:
					        ref = distance_matrix[row_number][column]
					        our_column = column
			    except KeyError:
				    pass
		    if our_column not in our_list:
			    our_list.append(our_column)
	    for row_num in range(1,distance_matrix.shape[0]):
		    try:
		        if row_num not in our_list:
			        raw_train_set.drop(row_num)
		    except KeyError:
			    pass
	    return raw_train_set


    def do_cnn(self, train_data, test_data, data_frame, k):
	    distance_matrix = self.get_distance_matrix(data_frame)
	    train_data = self.get_condensed_training_set(train_data)
	    predictions = self.knn_predict(train_data, test_data, k, distance_matrix)
	    return predictions





# EXECUTE SCRIPT


if __name__ == '__main__':

	print('running condensed knn...')
	condensed_knn = CondensedKNN()

	data_api_impl = DataApi('../../data/')
	wine_data = data_api_impl.get_raw_data_frame('abalone')
	print(wine_data)
	distance = condensed_knn.get_condensed_training_set(wine_data)
	print("BREAK")
	print(distance)



	#print(distance)


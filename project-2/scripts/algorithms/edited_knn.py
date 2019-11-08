#!/usr/bin/env python3


# IMPORTS


import sys
sys.path.append('../data_api')
sys.path.append('../preprocessing')
sys.path.append('../algorithms')
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
    This class handles all things edited knn. It inherits from the parent KNN class,
    in order to reuse the do_knn() method implemented in the KNN class.
'''


class EditedKNN(KNN):


    def __init__(self):
        KNN.__init__(self)
        self.DEBUG = True
        self.data_api_impl = DataApi('../../data/')

    def enn(self, train_data, test_data, data_frame, k):
      #features_full_df = data_frame.loc[:, data_frame.columns != 'CLASS']
      train_data = train_data.reset_index(drop=True)
      features_train_df = train_data.loc[:, data_frame.columns != 'CLASS']
      #features_test_df = test_data.loc[:, data_frame.columns != 'CLASS']

      #full_distance_matrix = self.get_distance_matrix(features_full_df)
      train_distance_matrix = self.get_distance_matrix(features_train_df)
      #test_distance_matrix = self.get_distance_matrix(features_test_df)

      #do some function call here that will return the edited training set
      edited_train_set = self.get_edited_training_set(train_data, train_distance_matrix, k)

      print("Edited Training Set: ")
      print(edited_train_set)
      print("Original Training Set Size: ")
      print(train_data.shape[0])
      print("Edited Training Set Size: ")
      print(edited_train_set.shape[0])

    def get_edited_training_set(self, train_data, distance_matrix, k):
      #while change is large (define change)
      #define change as number of edits (like number of dropped rows)
      #if it is the same after two consecutive loops, then break the while loop
        #loop through training data and use knn() to reduce the training set

      edited_train_set = train_data.copy()
      #features_train_df = edited_train_set.loc[:, train_data.columns != 'CLASS']

      number_of_edits = 100
      number_of_edits_previous = 1000
      loopcounter = 0
      while(number_of_edits_previous - number_of_edits > 5):
        number_of_edits_previous = number_of_edits
        number_of_edits = 0
        for data_point in edited_train_set.index:
          print("Running knn...")
          knn = self.knn(data_point, edited_train_set.loc[:, edited_train_set.columns != 'CLASS'], distance_matrix, k)
          print("Now running the class estimation...")
          print(edited_train_set)
          class_estimate = self.return_classification_estimation(edited_train_set, knn)
          print("Class Estimate: ")
          print(class_estimate)
          print("Actual Class: ")
          print(edited_train_set.loc[data_point]['CLASS'])
          if class_estimate == edited_train_set.loc[data_point]['CLASS']:
            continue
          else:
            print("Dropping a row: " + str(data_point))
            edited_train_set = edited_train_set.drop(data_point, axis=0)
            number_of_edits += 1

          print(loopcounter)
          print("Number of Edits: ")
          print(number_of_edits)
          print("Number of Previous Edits: ")
          print(number_of_edits_previous)
        loopcounter += 1
        print("Number of While Loops: ")

      return edited_train_set.reset_index(drop=True)



# EXECUTE SCRIPT


if __name__ == '__main__':

    print('running edited knn...')
    edited_knn = EditedKNN()

    data_api_impl = DataApi('../../data/')
    cross_validator_impl = CrossValidator()
    preprocessor_impl = Preprocessor()

    wine_data = data_api_impl.get_raw_data_frame('segmentation')
    prep_wine_data = preprocessor_impl.preprocess_raw_data_frame(wine_data, 'segmentation')



    wine_data_train_set = cross_validator_impl.get_training_set(prep_wine_data, test_set_number=3)
    print('wine_data_train_set.shape: ' + str(wine_data_train_set.shape))

    wine_data_test_set = cross_validator_impl.get_test_set(prep_wine_data, test_set_number, indexes_list)

    edited_knn.enn(wine_data_train_set, wine_data_test_set, prep_wine_data, k)

    edited_train_set = edited_knn.get_edited_training_set(wine_data_train_set, k=25)
    print('edited_train_set.shape: ' + str(edited_train_set.shape))

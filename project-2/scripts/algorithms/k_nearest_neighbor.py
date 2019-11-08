#!/usr/bin/env python3


# IMPORTS


import sys
sys.path.append('../data_api')
sys.path.append('../utilities')
sys.path.append('../cross_validator')
sys.path.append('../preprocessing')
sys.path.append('../../../data')

from data_api import DataApi
from distance_functions import DistanceFunctions
from cross_validator import CrossValidator
from preprocessor import Preprocessor

import time

import pandas as pd
import numpy as np
import statistics as stats

from random import shuffle
from operator import itemgetter
from statistics import StatisticsError
from scipy.spatial.distance import pdist, squareform


# CLASS

'''
    This class handles all things k nearest neighbor.
    TODO: make sure list of nearest neighbors is correct
'''


class KNN:


    def __init__(self):
        self.DEBUG = True
        self.VERBOSE = False


        self.data_api_impl = DataApi('../../../data/')
        self.data_set = None

        self.CLASSIFICATION = True
        self.REGRESSION = False

        self.algorithm_name = None

    def knn(self, data_point_index, train_data, distance_matrix, k):

      print("Train Data Indexes: ")
      print(train_data.index)
      print("Columns: ")
      print(train_data.columns)
      #distance_matrix = knn_impl.get_distance_matrix(train_data)
      print("Distance Matrix of Training Data: ")
      print(distance_matrix)
      print("Size of Distance Matrix: ")
      print(len(distance_matrix[100]))
      print("Data Point Index: ")
      print(data_point_index)

      knn = []
      knn_dist = [100]

      print("Train Data .index: ")
      print(train_data.index)

      for distance_index in train_data.index:
        #print("We are on loop number: " + str(distance_index))
        for neighbor in range(len(knn_dist)):
          if distance_matrix[data_point_index][distance_index] < knn_dist[neighbor] and distance_index is not data_point_index:
            knn.insert(neighbor, distance_index)
            knn_dist.insert(neighbor, distance_matrix[data_point_index][distance_index])
            break

          if len(knn) > k:
            knn = knn[0:k]
            #print(knn)

      # print(knn[0:k])

      # train_data_knn = []
      # train_data_indexes = train_data.index
      # for dist_matrix_index in range(k):
      #   train_data_knn.append(train_data_indexes[knn[dist_matrix_index]])

      return knn[0:k]

    def return_classification_estimation(self, data_frame, indexes):

      classes = []
      print("Indexes: ")
      print(indexes)
      for index in indexes:
        # print(index)
        print("This is a row of a nearest neighbor: ")
        print(data_frame.loc[index]['CLASS'])
        classes.append(data_frame.loc[index]['CLASS'])

      class_estimate = stats.mode(classes)


      return class_estimate


    def get_distance_matrix(self, data_frame):
        # for some reason this has to be done here even though it's done above...
        feature_vectors_df = data_frame.loc[:, data_frame.columns != 'CLASS']
        # get distance matrix (upper triangle) using distance metric
        distances = pdist(feature_vectors_df.values, metric='euclidean')
        # fill in lower triangle maintaining symmetry
        dist_matrix = squareform(distances)
        # return full distance matrix
        return dist_matrix


    # set data set name for context
    def set_data_set(self, data_set):
        self.data_set = data_set

        if self.data_set in ['abalone', 'car', 'segmentation']:
            # CLASSIFICATION data sets
            self.CLASSIFICATION = True
            self.REGRESSION = False
        elif self.data_set in ['machine', 'forestfires', 'wine']:
            # REGRESSION data set
            self.REGRESSION = True
            self.CLASSIFICATION = False


    # get data set name
    def get_data_set(self):
        return self.data_set


    # set algorithm name for context
    def set_algorithm_name(self, algorithm_name):
        self.algorithm_name = algorithm_name

# EXECUTE SCRIPT


if __name__ == '__main__':

    print('\nk nearest neighbor...\n')

    data_api_impl = DataApi('../../data/')
    cross_validator_impl = CrossValidator()
    preprocessor_impl = Preprocessor()

    knn_impl = KNN()

    segmentation_data = data_api_impl.get_raw_data_frame('segmentation')
    segmentation_data_preproc = preprocessor_impl.preprocess_raw_data_frame(segmentation_data, "segmentation")

    distance_matrix = knn_impl.get_distance_matrix(segmentation_data_preproc)
    print("Segmentation Data Preprocessed: ")
    print(segmentation_data_preproc)
    print("--------------------------------------------------------------------------------------")
    knn = knn_impl.knn(10, segmentation_data_preproc, distance_matrix, 5)
    print("Returned KNN: ")
    print(knn)
    class_estimate = knn_impl.return_classification_estimation(segmentation_data_preproc, knn)
    print("Class Estimate: ")
    print(class_estimate)

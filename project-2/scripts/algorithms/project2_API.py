#!/usr/bin/env python3


# IMPORTS


import sys
sys.path.append('../algorithms')

from data_api import DataApi
from edited_knn import EditedKNN
from condensed_knn import CondensedKNN
from k_means_clustering import KMeansClustering
from k_medoids_clustering import KMedoidsClustering

import math
import random
import pandas as pd


# CLASS

'''
    This class handles all things P2 API.
'''


class P2API():


    '''
    CONSTRUCTOR

    args:
        - data_set_name: name of data set used in this context
        - algorithm_name: name of algorithm used in this context
    '''
    def __init__(self, data_set_name, algorithm_name):
        self.DEBUG = False

        self.data_set_name = data_set_name
        self.algorithm_name = algorithm_name

        # algorithm implementations
        self.enn_impl = EditedKNN()
        self.enn_impl.set_data_set(data_set_name)
        self.enn_impl.set_algorithm_name(algorithm_name)

        # TODO: change the constuctors for the P2 algorithms so they can optionally
        # take in data_set_name and algorithm_name to clean up this code here

        self.cnn_impl = CondensedKNN()
        self.cnn_impl.set_data_set(data_set_name)
        self.cnn_impl.set_algorithm_name(algorithm_name)

        self.kmeans_knn_impl = KMeansClustering()
        self.kmeans_knn_impl.set_data_set(data_set_name)
        self.kmeans_knn_impl.set_algorithm_name(algorithm_name)

        self.k_medoids_clustering_impl = KMedoidsClustering()
        self.k_medoids_clustering_impl.set_data_set(data_set_name)
        self.k_medoids_clustering_impl.set_algorithm_name(algorithm_name)


    '''
    get edited training set for use in RBF network config

    INPUT:
        - train_data: training data to edit
        - data_frame: full data frame
        - k: value of k parameter in base knn

    OUTPUT:
        - return edited training data given raw training data and k
    '''
    def get_enn_for_rbf_network(self, train_data, data_frame, k):
        if self.DEBUG:
            print('get_enn_for_rbf_network: train_data.shape: %s' % str(train_data.shape))
            print('get_enn_for_rbf_network: data_frame.shape: %s' % str(data_frame.shape))

        return self.enn_impl.get_enn_for_rbf_network(train_data, data_frame, k)


    '''
    get condensed training set for use in RBF network config

    INPUT:
        - train_data: training data to condense

    OUTPUT:
        - return condensed training data given raw training data
    '''
    def get_cnn_for_rbf_network(self, train_data):
        return self.cnn_impl.get_condensed_training_set(train_data)


    '''
    get kmeans centroids for use in RBF network config

    INPUT:
        - train_data: training data to generate k centroids from
        - data_frame: full data frame
        - k: value of k parameter in base knn

    OUTPUT:
        - return kmeans centroids data given raw training data
    '''
    def get_kmeans_centroids_for_rbf_network(self, train_data, data_frame, k):
      return self.kmeans_knn_impl.get_centroids_for_rbf_network(train_data, data_frame, self.data_set_name, k)


    '''
    get kmedoids medoids for use in RBF network config

    INPUT:
        - train_data: training data to get k medoids from
        - data_frame: full data frame
        - k: value of k parameter in base knn

    OUTPUT:
        - return kmedoids medoids data given raw training data
    '''
    def get_kmedoids_medoids_for_rbf_network(self, train_data, data_frame, k):
      return self.k_medoids_clustering_impl.get_kmedoids_medoids_for_rbf_network(train_data, data_frame, k)



# EXECUTE SCRIPT


# if we run this script directly, the following code below will execute
if __name__ == "__main__":

    print('\nrunning P2API...')

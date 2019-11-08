#!/usr/bin/env python3


# IMPORTS


import sys
sys.path.append('../data_api')
sys.path.append('../../../data')

from data_api import DataApi
from k_nearest_neighbor import KNN
from edited_knn import EditedKNN
import math
from random import randint
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import numpy as np
import statistics as stats


# CLASS

'''
    This class handles all things k medoids clustering...
'''


class KMedoidsClustering(KNN):


    def __init__(self):
        KNN.__init__(self)
        self.DEBUG = False
        self.data_api_impl = DataApi('../../data/')
        self.enn_impl = EditedKNN()


    '''
    perform k-medoids-clustering against full_data_frame using k value as parameter

    INPUT:
      - full_data_frame: full data set
      - k: value for parameter k, i.e. the number of clusters to partition the data set into

    OUTPUT:
      - data structure holding k clusters and all data contained within each cluster
    '''
    def cluster(self, training_set, preprocessed_data_frame, parameter):

        #data_frame = full_data_frame.loc[:, full_data_frame.columns != 'CLASS']
        self.enn_impl.set_data_set(self.get_data_set())
        dist_matrix = self.get_distance_matrix(preprocessed_data_frame.loc[:, preprocessed_data_frame.columns != 'CLASS'])
        edited_data_frame = self.enn_impl.get_edited_training_set(training_set, dist_matrix, parameter)

        #if self.DEBUG:


        #data_frame = full_data_frame.loc[:, full_data_frame.columns != 'CLASS']

      #randomly choose k points
        #find the max number of data points
        print("Size of Edited Data Frame: ")
        print(edited_data_frame.shape[0])
        maxRange = edited_data_frame.shape[0]
        #print(maxRange)
        #this list holds the indices of the medoids in relation to the distance matrix
        medoid_indices = []
        #calculate k random numbers from (0, max data points)
        while len(medoid_indices) < parameter:
          randIndex = randint(0, maxRange - 1)
          #make sure we don't generate duplicate indicies
          if randIndex not in medoid_indices:
            medoid_indices.append(randIndex)

        # print("Medoid Indices: ")
        # print(medoid_indices)

        #this 2D list stores a list of indexes for points in a cluster
          #the first dimension corresponds to the index of medoid_indices
          #the second dimension stores a list of indicies of points in relation to the distance matrix
          #please note that the clusters[medoid_index][0] stores the lowest known score of distances
        clusters = []
        for x in range(parameter):
          for y in range(2):
            if y is 0:
              clusters.append([0])
            else:
              clusters[0].append(medoid_indices[x])
        #clusters = [[0 for x in range(k)] for y in range(edited_data_frame.size)] #[0:k-1][0:clusterSize]

        #some point to be placed in a cluster
        for x in range(len(dist_matrix)):
          #set this large so it will be overwritten
          smallest_distance = 1000000
          #this stores which medoid set we'll write to
          medoid_index = 0
          #we compare the distance from this point to each medoid point
          for y in range(len(medoid_indices)):
            #if we have a smaller distance, then save that cluster and the distance for future comparison
            if dist_matrix[x][y] < smallest_distance:
              smallest_distance = dist_matrix[x][y]
              medoid_index = y
          #This actually holds the lowest score for the current randomly generated medoid
          print("This is k: " + str(parameter))
          print("This is medoid indices: " )
          print(medoid_indices)
          print("This is clusters: ")
          print(clusters)
          print("This is the medoid index: " + str(medoid_index))
          clusters[medoid_index][0] += smallest_distance
          #append the index of the point we are placing in a cluster
          clusters[medoid_index].append(x)

        #we must now find the best fit point to use as the final medoid set from each cluster

        #save the initial medoids because they are not in the clusters data structure
        initial_medoids = medoid_indices

        #go cluster by cluster
        print("The number of clusters we have is: " + str(len(clusters)))
        print(clusters)


        for cluster in range(len(clusters)):
          #zero out score
          score = 0
          #now go point by point in the cluster
          for potential_medoid in range(2, len(clusters[cluster]) - 2):
            #compare it to each other point in the cluster
            if potential_medoid is not medoid_indices[cluster]:
              #print("The number of potential medoids of this cluster are: " + str(len(clusters[cluster])))
              for cluster_point in range(2, len(clusters[cluster]) - 2):
                #increment the score as a sum total of all the distances
                score += dist_matrix[potential_medoid][cluster_point]
              #add the score of the initial medoids because they are actually not in the clusters data structure

              score += dist_matrix[potential_medoid][initial_medoids[cluster]]
              #if the score is lower, it's a better candidate
              if score < clusters[cluster][0]:
                #set the new potential medoid as the medoid for the cluster
                medoid_indices[cluster] = potential_medoid
                #save its score
                clusters[cluster][0] = score

        #create new data frame to store subset
        training_set_df = pd.DataFrame()
        #put the medoid rows in the data frame from the full_data_frame
        print("The medoid indices: ")
        print(medoid_indices)
        print("The edited data frame: ")
        print(edited_data_frame)
        for x in medoid_indices:
          training_set_df = training_set_df.append(edited_data_frame.loc[x])
        #return to sender
        print(training_set_df)
        return training_set_df
        #return self.do_knn(training_set_df, test_set, preprocessed_data_frame, parameter)


    def get_kmedoids_medoids_for_rbf_network(self, train_data, data_frame, k):
        return self.cluster(train_data, data_frame, k)

    # def get_distance_matrix(self, data_frame):
    #     # for some reason this has to be done here even though it's done above...
    #     feature_vectors_df = data_frame.loc[:, data_frame.columns != 'CLASS']
    #     # get distance matrix (upper triangle) using distance metric
    #     distances = pdist(feature_vectors_df.values, metric='euclidean')
    #     # fill in lower triangle maintaining symmetry
    #     dist_matrix = squareform(distances)
    #     # return full distance matrix
    #     return dist_matrix


# EXECUTE SCRIPT


if __name__ == '__main__':

    print('k medoids clustering...')
    k_medoids_clustering_impl = KMedoidsClustering()

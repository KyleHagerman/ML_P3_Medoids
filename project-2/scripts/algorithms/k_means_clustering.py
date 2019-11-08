#!/usr/bin/env python3


# IMPORTS


import sys
sys.path.append('../data_api')
sys.path.append('../preprocessing')
sys.path.append('../cross_validator')
sys.path.append('../algorithms')
sys.path.append('../utilities')
sys.path.append('../../../data')

from data_api import DataApi
from preprocessor import Preprocessor
from cross_validator import CrossValidator
from distance_functions import DistanceFunctions
from k_nearest_neighbor import KNN
from utilities import Utilities

import time
import math
import random
import numpy as np
import pandas as pd
import statistics as stats
from scipy.spatial.distance import pdist, squareform


# CLASS

'''
    This class handles all things k means clustering...
'''


class KMeansClustering(KNN):


    def __init__(self):
        KNN.__init__(self)
        self.DEBUG = True
        self.VERBOSE = False
        self.data_api_impl = DataApi('../../data/')
        self.utilities_impl = Utilities()
        self.distance_functions_impl = DistanceFunctions()

        # threshold for clustering convergence
        # stop iterating when differences between consecutive centroids is smaller than this
        self.CONVERGENCE_THRESHOLD = 0.25
        # maximum clustering iterations allowed before returning answer
        self.MAX_ITERATIONS = 5
        #self.MAX_CLUSTER_TIME = 5 # minutes


    '''
    perform k-means-clustering against full_data_frame using k value as parameter

    INPUT:
        - data_set_name: name of data set to cluster
        - train_data: training data set to cluster
        - k: value for parameter k, i.e. the number of clusters to partition the data set into

    OUTPUT:
        - tuple:
            - index 1: list of cluster keys representing the cluster each data point belongs to
            - index 2: centroids dataframe for k centroids (without class values)
    '''
    def cluster(self, data_set_name, train_data, k):

        print('\nk means clustering with k: ' + str(k))

        # get list of column labels for data set, not including the CLASS column label
        data_column_labels = self.data_api_impl.get_column_labels(data_set_name, include_class=False)
        # get training data without class column
        train_data = train_data.loc[:, train_data.columns != 'CLASS']

        # randomly generate k initial centroids from training data
        centroids = self.generate_initial_centroids(train_data, k)

        #print('generated initial centroids')

        if self.DEBUG:
            print('centroids:')
            print(centroids)

        if not isinstance(centroids, pd.DataFrame):
            # convert list of centroids to data frame using same column labels
            centroids_df = pd.DataFrame.from_records(centroids, columns=data_column_labels)
            #print('created centroids_df from records from centroids')
        else:
            centroids_df = centroids

        # combine the centroids_df and train_data dataframes into one frame with centroids first
        centroids_and_data_df = centroids_df.append(train_data, ignore_index=True)

        #print('created centroids_and_data_df')

        # get distance matrix - distance from every training point to every centroid
        distance_from_centroids = self.get_distance_matrix(centroids_and_data_df)

        #print('calculated distance matrix for centroids_and_data_df')

        if self.DEBUG and self.VERBOSE:
            print('cluster: train_data.shape: ' + str(train_data.shape))
            print('cluster: k: ' + str(k))
            print('cluster: number of centroids: ' + str(len(centroids)))
            print('centroids_and_data_df:')
            print(centroids_and_data_df)
            print('cluster: centroids_and_data_df.shape: ' + str(centroids_and_data_df.shape))
            print('cluster: distance_from_centroids.shape: ' + str(distance_from_centroids.shape))

        cluster_assignments = []
        iteration_count = 1

        # initial centroids
        previous_centroids = centroids_and_data_df.iloc[:k, :]
        # set initial centroids diffs to maximum number sizes
        centroids_diff = [sys.maxsize for i in range(k)]

        cluster_start_time = time.time()

        #print('right before clustering...')

        while not self.threshold_reached(centroids_diff) and iteration_count < self.MAX_ITERATIONS:

            print('clustering... iteration: ' + str(iteration_count))

            new_cluster_assignments = []

            # for each training point in the training data (start indexing at k)
            for instance_idx in range(k, centroids_and_data_df.shape[0]):
                data_point = centroids_and_data_df.iloc[instance_idx, :]
                # get list of distances from instance to each centroid
                idx_distances = np.array(distance_from_centroids[instance_idx][:k])
                # get index of centroid with least distance from instance
                closest_centroid_idx = np.argmin(idx_distances)
                # map instance index to centroid index (0 - k)
                new_cluster_assignments.append(closest_centroid_idx)

            #print('calculated new cluster assignments')

            cluster_assignments = new_cluster_assignments
            updated_centroids = []

            # update each centroid to mean of all points assigned to that centroid
            for centroid_idx in range(k):
                #print('updating mean for centroid: ' + str(centroid_idx))

                cluster_points = pd.DataFrame(columns=data_column_labels)
                np_cluster_assignments = np.array(cluster_assignments)
                # get list of indexes for all instances assigned to given centroid index
                idxs_for_cluster_val = np.where(np_cluster_assignments == centroid_idx)[0]

                for idx in idxs_for_cluster_val:
                    # add point to cluster points dataframe, add k to row index to skip centroid points
                    # NOT EFFICIENT - CHANGE THIS SO IT IS FASTER
                    cluster_points = cluster_points.append(centroids_and_data_df.iloc[idx+k, :])

                #print('built cluster points list')

                if self.DEBUG and self.VERBOSE:
                    print('centroid_idx: ' + str(centroid_idx))
                    print('idxs_for_cluster_val:' + str(idxs_for_cluster_val))
                    print('points in cluster ' + str(centroid_idx) + ': ' + str(cluster_points.shape))
                    print('cluster_points:')
                    print(cluster_points)

                avg_centroid = None

                # if there are points that were assigned to the cluster (centroid index)
                if cluster_points.shape[0] > 0:
                    avg_centroid = self.get_avg_centroid(cluster_points)
                    #print('got average centroid for idx: ' + str(centroid_idx))
                    if self.DEBUG and self.VERBOSE:
                        print('avg_centroid:')
                        print(avg_centroid)

                # do not update the centroid if any of the values in the centroid are nan
                if avg_centroid is not None and not np.isnan(np.array(avg_centroid)).any():
                    updated_centroids.append(avg_centroid)
                    #print('appended avg_centroid to updated_centroids list')
                else:
                    if self.DEBUG and self.VERBOSE:
                        print('ERROR: avg_centroid is none!')
                        print('bad cluster points:')
                        print(cluster_points)
                        print('avg_centroid:')
                        print(avg_centroid)
                    # keep previous centroid if the average centroid is still null (no points in cluster)
                    updated_centroids.append(list(previous_centroids.iloc[centroid_idx, :].values))
                    #print('appended previous centroid to updated_centroids list')

            # update centroids dataframe using list of updated centroids representing new average centroids
            updated_centroids_df = pd.DataFrame.from_records(updated_centroids, columns=data_column_labels)

            #print('created updated_centroids_df from records of updated_centroids')

            # update centroids in reference data frame (the one that contains the points too)
            for row_num in range(updated_centroids_df.shape[0]):
                centroids_and_data_df.iloc[row_num, :] = updated_centroids_df.iloc[row_num, :]

            #print('updated centroids in reference data frame centroids_and_data_df')

            # update distance matrix using new centroids for distance calculations
            distance_from_centroids = self.get_distance_matrix(centroids_and_data_df)

            #print('updated distance matrix using new centroids')

            # calculate distance between pairs of previous/new centroids to see if we've satisfied the threshold
            centroids_diff = self.get_centroids_diff(previous_centroids, updated_centroids_df)

            #print('calculated centroids diff')

            # BUG: fix the issue where the first centroid diff is always zero
            if centroids_diff[0] == 0:
                #iteration_count = iteration_count - 1 # remove this line?
                #print('decremented iteration_count to: ' + str(iteration_count) + ', because centroids_diff[0] == 0')
                #print('centroids_diff[0] == 0 !!!')
                iteration_count = iteration_count + 1
                continue # workaround for now
            else:
                print('\nclustering iteration: ' + str(iteration_count + 1))
                print('centroids_diff: ' + str(centroids_diff))

            # update previous centroids dataframe to updated centroids dataframe
            previous_centroids = updated_centroids_df
            #print('updated previous centroids using updated_centroids_df')
            iteration_count = iteration_count + 1

        print('cluster returning: %s, %s, %s' % (str(len(cluster_assignments)), \
                            str(updated_centroids_df.shape), str(iteration_count)))

        # return a tuple containing the final list of cluster assignments and the final centroids
        return (cluster_assignments, updated_centroids_df, iteration_count - 1)


    '''
    generate initial cluster centroids with random values in min/max range for each column

    INPUT:
        - data_frame: data to generate centroids for
        - k: k param value, i.e. number of centroids to generate

    OUTPUT:
        - list of centroid points with same dimensionality as regular data points
    '''
    def generate_initial_centroids(self, data_frame, k):
        '''
        # RANDOM GENERATION APPROACH
        centroids = []
        # get min/max values for each column (the bounds of the values for each column)
        column_bounds = self.utilities_impl.get_column_bounds(data_frame)
        num_cols = len(column_bounds)
        for centroid_index in range(k):
            centroid = []
            for col_index in range(num_cols):
                min_max_bounds = column_bounds[col_index]
                # randomly generate value in min/max range for each attribute
                centroid.append(random.uniform(min_max_bounds[0], min_max_bounds[1]))
            centroids.append(centroid)
        # return list of centroid points
        return centroids
        '''
        # RANDOM POINTS APPROACH
        indexes = random.sample(range(data_frame.shape[0]), k)
        # BUG: change this so it doesn't throw the pandas error in the log
        return data_frame.reindex(indexes)
        

    # return boolean indicating whether the centroid diff threshold has been reached
    def threshold_reached(self, centroids_diff):
        np_diffs = np.array(centroids_diff)
        # workaround for bug causing centroids_diff to be all zeros
        if np_diffs is None or np_diffs[0] == 0:
            return False
        # return boolean indicating whether any centroid diffs are greater than threshold
        return not list(np_diffs[np_diffs > self.CONVERGENCE_THRESHOLD])


    '''
    get average centroid from all points assigned to given cluster

    INPUT:
        - cluster_points: dataframe consisting of all points assigned to cluster

    OUTPUT:
        - centroid where each column is average value of respective column in cluster_points
    '''
    def get_avg_centroid(self, cluster_points):
        avg_col_vals = []
        # for each column in dataframe representing all points assigned to cluster
        for column_label, _ in cluster_points.items():
            column_vals = cluster_points.loc[:, column_label].values
            column_vals = [float(val) for val in column_vals]
            # calculate average column value and append to list
            avg_col_vals.append(stats.mean(column_vals))
            if self.DEBUG and self.VERBOSE:
                print('column_label: ' + str(column_label))
                print('len(column_vals): ' + str(len(column_vals)))
                print('column_vals: ')
                print(column_vals)
                print('avg_column_vals: ' + str(stats.mean(column_vals)))
        # return average centroid as list of average values for each column
        return avg_col_vals


    '''
    get diff between centroids from iteration n and iteration n+1
    '''
    def get_centroids_diff(self, previous_centroids, updated_centroids_df):
        if self.DEBUG and self.VERBOSE:
            print('previous_centroids:')
            print(previous_centroids)
            print('updated_centroids_df:')
            print(updated_centroids_df)
        centroid_diffs = []
        # for each centroid (instance) in the previous centroids dataframe
        for row_num in range(previous_centroids.shape[0]):
            prev_row = previous_centroids.iloc[row_num, :]
            updated_row = updated_centroids_df.iloc[row_num, :]
            # calculate euclidean distance between previous and updated centroid instance
            diff_dist = self.distance_functions_impl.get_euclidean_distance(prev_row, updated_row)
            centroid_diffs.append(diff_dist)
        # return list containing distances between each corresponding pair of centroids
        return centroid_diffs


    '''
    evaluate clustering - show counts
    '''
    def evaluate_clustering(self, data, clustering_assignments, k):
        print('\nCLUSTERING EVALUATION:')
        np_cluster_assignments = np.array(cluster_assignments)
        for centroid_idx in range(k):
            print('centroid_idx: ' + str(centroid_idx))
            freqs = {}
            idxs_for_cluster_val = np.where(np_cluster_assignments == centroid_idx)[0]
            for idx in idxs_for_cluster_val:
                if idx in data.index:
                    actual_class = str(data.loc[idx, 'CLASS'])
                    if actual_class in freqs:
                        freqs[actual_class] = freqs[actual_class] + 1
                    else:
                        freqs[actual_class] = 1
                else:
                    print('ERROR: ' + str(idx) + ' not in data.index!')
            print('freqs: ' + str(freqs))


    '''
    get distance matrix using pdist and squareform methods from scipy.spatial.distance

    INPUT:
        - data_frame: data frame we're working with

    OUTPUT:
        - distance matrix containing distances between every pair of points in data frame
    '''
    def get_distance_matrix(self, data_frame):
        # get distance matrix (upper triangle) using distance metric
        distances = pdist(data_frame.values, metric='euclidean')
        # fill in lower triangle maintaining symmetry
        dist_matrix = squareform(distances)
        # return full distance matrix
        return dist_matrix


    '''
    method for getting k cluster centroids using clustering output from cluster() method above
    the input 'centroids_data' is a dataframe that contains all the attribute value for the centroids
    this method is responsible for appending the corresponding class values to each centroid instance

    INPUT:
        - cluster_assignments - list of clustering assignments
        - centroids_data - centroids rows without class values

    OUTPUT:
        - data frame with k rows, representing k centroids
    '''
    def get_cluster_centroids(self, cluster_assignments, centroids_data, dataframe):
        if self.DEBUG and self.VERBOSE:
            print('get_cluster_centroids: unique cluster assignments: ' + str(set(cluster_assignments)))
            print('get_cluster_centroids: centroids_data.shape - BEFORE: ' + str(centroids_data.shape))
            print('get_cluster_centroids: dataframe.shape: ' + str(dataframe.shape))
            print('get_cluster_centroids: dataframe:')
            print(dataframe)

        # convert list of cluster assignments to np array to utilize np methods
        np_cluster_assignments = np.array(cluster_assignments)
        centroid_class_vals = []
        #print('len set cluster assignments: ' + str(len(set(cluster_assignments))))

        # for each unique cluster value (centroid index) that data points were assigned to
        for unique_cluster_val in set(cluster_assignments):
            # get dataframe row indexes that were assigned to the cluster
            val_idxs = np.where(np_cluster_assignments == unique_cluster_val)[0]
            #print('get_cluster_centroids: val_idxs assigned to unique_cluster_val: ' + str(unique_cluster_val) + ' --> ' + str(val_idxs))
            idx_class_vals = self.get_idx_class_vals(dataframe, val_idxs)
            #print('get_cluster_centroids: idx_class_vals for unique_cluster_val: ' + str(unique_cluster_val) + ' --> ' + str(idx_class_vals))
            highest_freq_class = self.utilities_impl.get_mode(idx_class_vals)
            print('highest_freq_class: ' + str(highest_freq_class))
            # append highest frequency class to list of centroid class values
            centroid_class_vals.append(highest_freq_class)

        #print('len centroid_class_vals: ' + str(len(centroid_class_vals)))
        #print('get_cluster_centroids: centroid_class_vals: ' + str(centroid_class_vals))

        # these values will not match if there were clusters that had no points assigned to them
        if len(centroid_class_vals) != centroids_data.shape[0]:
            # get list of all possible class values for given dataframe
            poss_class_vals = list(set(dataframe.loc[:, 'CLASS'].values))
            # randomly assign class values to missing clusters to make dimensions match
            # BUG: this shouldn't be necessary, handle missing clusters in a better way
            centroid_class_vals = self.handle_cluster_count_mismatch(\
                centroid_class_vals, centroids_data.shape[0], poss_class_vals)

        # append generated class values column to centroids dataframe (assigning class to each centroid)
        centroids_data['CLASS'] = centroid_class_vals

        if self.DEBUG and self.VERBOSE:
            print('get_cluster_centroids: centroids_data.shape - AFTER: ' + str(centroids_data.shape))
            print('get_cluster_centroids: centroids_data - AFTER:')
            print(centroids_data)

        # return complete centroids dataframe (now containing the corresponding class values for each centroid)
        return centroids_data


    # get class values from dataframe for all indexes in idxs arg
    def get_idx_class_vals(self, dataframe, idxs):
        class_vals = []
        for idx in idxs:
            row_data = dataframe.iloc[idx, :]
            class_vals.append(row_data['CLASS'])
        # return list of class values for idxs
        return class_vals


    # this shouldn't be necessary, but for now workaround by returning random class value
    def handle_cluster_count_mismatch(self, centroid_class_vals, expected_centroid_count, poss_class_vals):
        while len(centroid_class_vals) < expected_centroid_count:
            # BUG: this shouldn't be necessary, for now return random class value for missing clusters
            centroid_class_vals.append(random.choice(poss_class_vals))
        assert len(centroid_class_vals) == expected_centroid_count
        # return list of centroid class values with expected length
        return centroid_class_vals


    '''
    do full knn run through using k means clustering output as reduced data set for knn
    NOTE: this always sets k equal to the number of possible class values for the given data set

    INPUT:
        - train_data: training data that will be clustered
        - test_data: test data
        - dataframe: full dataframe
        - data_name: name of data set we're using
        - k: value for k parameter

    OUTPUT:
        - returns a dictionary where key is the instance index and value is a tuple: (prediction, actual)
    '''
    def cluster_do_knn(self, train_data, test_data, dataframe, data_name, k):
        # get number of possible class values for given data frame
        num_poss_class_vals = len(set(dataframe.loc[:, 'CLASS'].values))
        # k means cluster the training data to get list of final cluster assignments and resulting centroids
        cluster_assignments, centroids_data, iteration_count = self.cluster(data_name, train_data, k=num_poss_class_vals)

        '''
        would be cool if we did some cluster evaluation but this method isn't working for some reason...

        k_means_clustering_impl.evaluate_clustering(training_set, cluster_assignments, k=5)
        '''

        # get resulting centroids using highest frequency class value for each set of points in clusters
        centroids_data = self.get_cluster_centroids(cluster_assignments, centroids_data, train_data)
        # return a dictionary where key is the instance index and value is a tuple: (prediction, actual)

        print('K MEANS CLUSTERING CONVERGED. iterations: ' + str(iteration_count))

        return self.do_knn(centroids_data, test_data, dataframe, k)


    '''
    get cluster centroids for use in RBF network setup

    INPUT:
        - train_data: training data that will be clustered
        - dataframe: full dataframe
        - data_name: name of data set we're using
        - k: value of k parameter in base knn

    OUTPUT:
        - return cluster centroids for given training data and k value
    '''
    def get_centroids_for_rbf_network(self, train_data, dataframe, data_name, k):
        # get number of possible class values for given data frame
        num_poss_class_vals = len(set(dataframe.loc[:, 'CLASS'].values))
        # k means cluster the training data to get list of final cluster assignments and resulting centroids
        cluster_assignments, centroids_data, iteration_count = self.cluster(data_name, train_data, k=num_poss_class_vals)
        # get resulting centroids using highest frequency class value for each set of points in clusters
        centroids_data = self.get_cluster_centroids(cluster_assignments, centroids_data, train_data)
        # return a dictionary where key is the instance index and value is a tuple: (prediction, actual)
        print('K MEANS CLUSTERING CONVERGED. iterations: ' + str(iteration_count))
        return centroids_data



# EXECUTE SCRIPT


if __name__ == '__main__':

    print('k means clustering...')
    k_means_clustering_impl = KMeansClustering()

    data_api_impl = DataApi('../../data/')
    preprocessor_impl = Preprocessor()
    cross_validator_impl = CrossValidator()

    '''
    wine_data = data_api_impl.get_raw_data_frame('wine')
    prep_wine_data = preprocessor_impl.preprocess_raw_data_frame(wine_data, 'wine')
    '''

    abalone_data = data_api_impl.get_raw_data_frame('abalone')
    prep_abalone_data = preprocessor_impl.preprocess_raw_data_frame(abalone_data, 'abalone')

    print('\npossible classes: ' + str(list(set(abalone_data.loc[:, 'CLASS'].values))) + '\n')

    training_set, test_set = cross_validator_impl.get_cv_partitions(prep_abalone_data)[0]

    # get training set (full data frame - rows in test_set_index bucket)
    #training_set = cross_validator_impl.get_training_set(prep_abalone_data, test_set_number=3)
    print('\ntraining_set.shape: ' + str(training_set.shape))
    # get test set (rows in test_set_index bucket)
    #test_set = cross_validator_impl.get_test_set(prep_abalone_data, test_set_number=3)
    print('test_set.shape: ' + str(test_set.shape))

    print('SET: ' + str(set(abalone_data.loc[:, 'CLASS'].values)))
    num_poss_class_vals = len(set(abalone_data.loc[:, 'CLASS'].values))

    cluster_assignments, centroids_data = k_means_clustering_impl.cluster('abalone', training_set, k=num_poss_class_vals)

    #k_means_clustering_impl.evaluate_clustering(training_set, cluster_assignments, k=5)
    
    centroids_data = k_means_clustering_impl.get_cluster_centroids(cluster_assignments, centroids_data, training_set)

    print('\ncentroids_data main:')
    print(centroids_data)


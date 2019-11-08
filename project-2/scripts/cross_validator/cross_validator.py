#!/usr/bin/env python3


# IMPORTS


import sys
sys.path.append('../data_api')
sys.path.append('../preprocessing')
sys.path.append('../../../data')

from data_api import DataApi
from preprocessor import Preprocessor

import pandas as pd

import random


# CLASS


'''
    This class handles doing 10-fold cross validation. It uses a list of indexes to partition
    and shuffle the data into 10 partitions of roughly equal size. The class has one method to
    get a training set given a data frame and a test set number, and another method to get a
    test set given a data frame and a test set number.
'''


class CrossValidator:


    def __init__(self):
        self.DEBUG  = False
        # set folds equal to the number of cross validation folds
        self.folds = 10


    # return shuffled list of indexes (with values 1 - #folds) used to partition data for cross validation
    def get_indexes_list(self, data_frame):
        indexes_list = []
        # calculate partition size
        slice_count = (data_frame.shape[0] - (data_frame.shape[0] % self.folds)) / self.folds
        slice_spot = 1
        counter = 0
        for i in range(data_frame.shape[0]):
            counter += 1
            indexes_list.append(slice_spot)
            if counter == slice_count and slice_spot == self.folds:
                pass
            elif counter == slice_count:
                slice_spot += 1 # need to double check how many rows are in
                counter = 0

        # randomly shuffle list of indexes so rows will randomly get assigned to partitions
        random.shuffle(indexes_list)
        # return list of randomly shuffled indexes
        return indexes_list


    # return training set given a data frame, test set number, and indexes list
    def get_training_set(self, data_frame, test_set_number, indexes_list):
        data_copy = data_frame.copy()
        test_set_list = []
        for i in range(data_frame.shape[0]):
            if indexes_list[i] == test_set_number:
                test_set_list.append(i)
            else:
                continue
        for drop_column in test_set_list:
            try:
                # pandas will throw error if column does not exist
                data_copy.drop(drop_column, inplace=True)
            except:
                continue
        # return training set data frame
        return data_copy


    # return test set given a data frame, test set number, and indexes list
    def get_test_set(self, data_frame, test_set_number, indexes_list):
        data_copy = data_frame.copy()
        test_set_list = []
        for i in range(data_frame.shape[0]):
            if indexes_list[i] == test_set_number:
                continue
            else:
                test_set_list.append(i)
        for drop_column in test_set_list:
            try:
                # pandas will throw error if column does not exist
                data_copy.drop(drop_column, inplace=True)
            except:
                continue
        # return test set data frame
        return data_copy


    '''
    return dictionary where key is 'test_set_number' and value is tuple:
        - tuple is (train_data, test_data) for that test set number
    '''
    def get_cv_partitions(self, data_frame):
        # get list of data frame indexes (test_instance.name vals)
        data_frame_indexes = list(data_frame.index.values)

        if self.DEBUG:
            print('\ndata_frame_indexes:')
            print(data_frame_indexes)

        # randomly shuffle list of data frame indexes
        random.shuffle(data_frame_indexes)

        if self.DEBUG:
            print('\n\nshuffled data_frame_indexes:')
            print(data_frame_indexes)

        num_df_indexes = len(data_frame_indexes)
        # calculate partition size (test set size)
        partition_size = int(num_df_indexes / self.folds)

        if self.DEBUG:
            print('num_df_indexes: ' + str(num_df_indexes))
            print('partition_size: ' + str(partition_size))

        cv_partitions = {}

        for fold in range(self.folds):
            test_data_indexes = data_frame_indexes[fold*partition_size : fold*partition_size + partition_size]
            train_data_indexes = [df_idx for df_idx in data_frame_indexes if df_idx not in test_data_indexes]

            if self.DEBUG:
                print('CV: train_data_indexes:')
                print(train_data_indexes)
                print('CV: test_data_indexes:')
                print(test_data_indexes)

            test_data = data_frame.loc[test_data_indexes, :]
            train_data = data_frame.loc[train_data_indexes, :]
            cv_partitions[fold] = (train_data, test_data)

        # return dictionary where key is 'test_set_number' and value is tuple: (train, test)
        return cv_partitions



# EXECUTE SCRIPT


if __name__ == "__main__":

    data_api_impl = DataApi('../../data/')
    cross_validator_impl = CrossValidator()
    preprocessor_impl = Preprocessor()

    abalone_data = data_api_impl.get_raw_data_frame('abalone')
    abalone_data = preprocessor_impl.preprocess_raw_data_frame(abalone_data, 'abalone')

    '''
    idx_list = cross_validator_impl.get_indexes_list(abalone_data)

    train_data = cross_validator_impl.get_training_set(abalone_data, 2, idx_list)
    test_data = cross_validator_impl.get_test_set(abalone_data, 2, idx_list)
    '''

    cv_partitions = cross_validator_impl.get_cv_partitions(abalone_data)

    '''
    for fold in cv_partitions:
        print('\nFOLD: ' + str(fold))
        print('train_data:')
        print(cv_partitions[fold][0])
        print('test_data:')
        print(cv_partitions[fold][1])
    '''

    print('len cv_partitions:')
    print(len(cv_partitions))
    print('cv_partitions keys:')
    print(str(list(cv_partitions.keys())))

    cv_1_train_data = cv_partitions[0][0]
    print('cv_partitions: 1: train_data:')
    print(cv_1_train_data)

    print('\nrow:')
    print(cv_1_train_data.iloc[2, :])

    '''
    train_data, test_data = cross_validator_impl.get_cross_validation_partition(abalone_data, test_set_number=5)
    '''

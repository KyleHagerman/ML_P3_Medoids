#!/usr/bin/env python3


# IMPORTS


import sys
# add following directories to class path
sys.path.append('../../data')
sys.path.append('../networks')
sys.path.append('../logging')
sys.path.append('../../../project-2/scripts/data_api')
sys.path.append('../../../project-2/scripts/preprocessing')
sys.path.append('../../../project-2/scripts/cross_validator')
sys.path.append('../../../project-2/scripts/utilities')

from data_api import DataApi
from preprocessor import Preprocessor
from cross_validator import CrossValidator
from neural_network import NeuralNetwork
from mlp_network import MLPNetwork
from rbf_network import RBFNetwork
from logger import Logger
from utils import Utils

import numpy as np


# CLASS

'''
    This class handles everything for configuring and running experiments.
'''

class ExperimentRunner:


    '''
    CONSTRUCTOR
    '''
    def __init__(self):
        # logger instance - VERBOSE level is highest (most verbose) level for logging
        self.logger = Logger('DEMO') # configure log level here

        # datalayer instance - read csv data files and convert into raw data frames
        self.datalayer = DataApi('../../data/')
        # preprocessor instance - everything for prerocessing data frames
        self.preprocessor = Preprocessor()
        # cross_validator instance - setup cross validation partitions
        self.cross_validator = CrossValidator()
        # utils instance - random things
        self.utils = Utils()


    # get average result given cross validation results dictionary
    def get_avg_result(self, cv_results):
        result_vals = []
        # for each cross validation partition, append result value to corresponding list
        for test_data_key in cv_results:
            test_result = cv_results[test_data_key]
            result_vals.append(test_result)

        # should always equal the value of the 'folds' variable in cross validator
        test_data_count = len(cv_results)
        # calculate average values
        avg_result = sum(result_vals) / test_data_count
        # return average result
        return avg_result


    '''
    get preprocessed data ready for consumption by experiment running logic

    INPUT:
        - data_set_name: name of data set to fetch data for

    OUTPUT:
        - preprocessed data frame - fully ready for experiment consumption
    '''
    def get_experiment_data(self, data_set_name):
        data = self.datalayer.get_raw_data_frame(data_set_name)
        self.logger.log('DEMO', 'data_set_name: \t%s\n' % str(data_set_name))
        self.logger.log('DEMO', 'raw data: \n\n%s, shape: %s\n' % (str(data), str(data.shape)))
        self.logger.log('DEMO', '----------------------------------------------------' \
                                    + '-----------------------------------------------\n')
        data = self.preprocessor.preprocess_raw_data_frame(data, data_set_name)
        self.logger.log('DEMO', 'preprocessed data: \n\n%s, shape: %s\n' % (str(data), str(data.shape)))
        self.logger.log('DEMO', '----------------------------------------------------' \
                                    + '-----------------------------------------------\n')
        return data


    '''
    run experiment

    INPUT:
        - data_set_name: name of data set to run experiment on
        - neural_network: instance of neural network to train/test with data
        - hyperparams: hyperparameters and corresponding values to use in experiment

    OUTPUT:
        - <void> - logs all the important stuff at DEMO level
    '''
    def run_experiment(self, data_set_name, neural_network, hyperparams):

        # LAYER ACTIVATION FUNCTION SPECIFICATION

        self.logger.log('DEMO', 'layer_activation_funcs: %s\n' % str(hyperparams["layer_activation_funcs"]))

        # DATA RETRIEVAL AND PREPROCESSING

        data = self.get_experiment_data(data_set_name)

        self.logger.log('DEMO', 'data_set_name: %s\n' % str(data_set_name))

        # CROSS VALIDATION PARTITIONING

        # get cross validation partitions for data
        cv_partitions = self.cross_validator.get_cv_partitions(data)

        # dictionary for storing accuracy results
        cv_results = {}
        # list of sizes of test sets used for getting average test set size
        test_data_sizes = []

        # NEURAL NETWORK TRAINING AND TESTING

        for partition in cv_partitions:
            # initialize key and corresponding nested dictionary in results dictionary
            test_data_key = 'test_data_' + str(partition)
            cv_results[test_data_key] = {}
            # get training set and test set for given cross validation partition
            train_data, test_data = cv_partitions[partition]
            test_data_sizes.append(test_data.shape[0]) # add number of rows in test set to test_set_sizes list

            # HANDLE RBF NETWORK P2 RESULTS

            if neural_network.network_name == 'RBF':
                # configure RBF network shape based on training data
                neural_network.configure_rbf_network(train_data, data, data_set_name, hyperparams["k"])

            # GRADIENT DESCENT

            # run gradient descent for given neural network instance
            test_result_vals = neural_network.train_gradient_descent(train_data, hyperparams, partition, test_data)

            self.logger.log('DEMO', ('accuracy_vals' if neural_network.CLASSIFICATION else 'error_vals') \
                + ' for partition %s: %s\n' % (str(partition+1), str(test_result_vals)), True)

            # append accuracy/error result of final gradient descent iteration to results dictionary
            cv_results[test_data_key] = test_result_vals[-1]

        # FINAL RESULTS (THE MODEL)

        self.logger.log('DEMO', '------------------------------------------------------------' \
                + ' TRAINING DONE ------------------------------------------------------------')

        self.logger.log('DEMO', 'trained network: weights --> \n\n%s, shapes: %s\n' \
            % (str(neural_network.weights), str(self.utils.get_shapes(neural_network.weights))), True)

        self.logger.log('DEMO', 'trained network: biases --> \n\n%s, shapes: %s\n' \
            % (str(neural_network.biases), str(self.utils.get_shapes(neural_network.biases))), True)

        self.logger.log('DEMO', 'data_set_name: %s\n' % str(data_set_name), True)

        self.logger.log('DEMO', 'trained network: AVERAGE ' \
            + ('ACCURACY' if neural_network.CLASSIFICATION else 'ERROR') + ' --> %s\n' \
            % str(self.get_avg_result(cv_results)), True)



# EXECUTE SCRIPT


if __name__ == '__main__':

    print('\nrunning Experiment from ExperimentRunner...\n')

    experiment_runner = ExperimentRunner()


    # DATA SET CONFIGURATION

    # CHANGE HERE: specify data set name
    data_set_name = 'segmentation'

    # NETWORK INSTANTIATION

    '''
    the implementation allows for an arbitrary number of inputs/outputs,
    but the networks must have logical i/o shapes based on each data set.

    note the following requirements for network shapes:

    CLASSIFICATION:
        - segmentation: 19 inputs, 7 outputs
        - car: 6 inputs, 4 outputs
        - abalone: 8 inputs, 28 outputs

    REGRESSION:
        - machine: 9 inputs, 1 output (regression)
        - forest fires: 12 inputs, 1 output (regression)
        - wine: 11 inputs, 1 output (regression)
    '''

    # CHANGE HERE: create MLP neural network instance
    #neural_network = MLPNetwork(data_set_name, [8, 2, 28]) # abalone mlp network
    #neural_network = MLPNetwork(data_set_name, [6, 10, 4]) # car mlp network
    #neural_network = MLPNetwork(data_set_name, [19, 10, 7]) # segmentation mlp network
    #neural_network = MLPNetwork(data_set_name, [9, 6, 3, 1]) # machine mlp network
    #neural_network = MLPNetwork(data_set_name, [12, 10, 1]) # forest fires mlp network
    #neural_network = MLPNetwork(data_set_name, [11, 7, 1]) # wine mlp network

    # CHANGE HERE: create RBF neural network instance
    #neural_network = RBFNetwork(data_set_name, [8, 0, 28], 'kmedoids_knn') # abalone rbf network
    #neural_network = RBFNetwork(data_set_name, [6, 0, 4], 'enn') # car rbf network
    neural_network = RBFNetwork(data_set_name, [19, 0, 7], 'kmedoids_knn') # segmentation rbf network
    #neural_network = RBFNetwork(data_set_name, [9, 0, 1], 'enn') # machine rbf network
    #neural_network = RBFNetwork(data_set_name, [12, 0, 1], 'enn') # forest fires rbf network
    #neural_network = RBFNetwork(data_set_name, [11, 0, 1], 'enn') # wine rbf network

    # HYPERPARAMETERS

    hyperparams = {}

    # CHANGE HERE: configure training parameters for gradient descent
    hyperparams['max_iterations'] = 10
    hyperparams['batch_size'] = 50
    hyperparams['eta'] = 3

    # CHANGE HERE: configure activation functions for each layer, options: ['sigmoid', 'relu', 'tanh']
    hyperparams['layer_activation_funcs'] = ['sigmoid' for layer_idx in range(len(neural_network.layer_sizes)-1)]
    #hyperparams["layer_activation_funcs"][-1] = 'sigmoid' # use sigmoid for output layer

    if neural_network.network_name == 'RBF':
        # DO NOT change this line here - this is not a config line, just used for better demo logging
        hyperparams['layer_activation_funcs'] = ['rbf', hyperparams['layer_activation_funcs'][-1]]

    # CHANGE HERE: configure whether momentum should be used in training
    hyperparams['use_momentum'] = False
    hyperparams['momentum_beta'] = 0.9 # commonly used value for momentum beta

    # RBF NETWORK SPECIFIC CONFIG

    # CHANGE HERE: k value for k nearest neighbor and variants from P2
    hyperparams['k'] = 10


    # RUN EXPERIMENT

    experiment_runner.run_experiment(data_set_name, neural_network, hyperparams)

#!/usr/bin/env python3


# IMPORTS


import sys
# add following directories to class path
sys.path.append('../../data')
sys.path.append('../data_api')
sys.path.append('../preprocessing')
sys.path.append('../cross_validator')
sys.path.append('../algorithms')
sys.path.append('../tuning')
sys.path.append('../results')

import pandas as pd

from data_api import DataApi
from preprocessor import Preprocessor
from cross_validator import CrossValidator
from k_nearest_neighbor import KNN
from edited_knn import EditedKNN
from k_means_clustering import KMeansClustering
from k_medoids_clustering import KMedoidsClustering
from parameter_tuner import ParameterTuner
from results import Results
from condensed_knn import CondensedKNN
from results_plotter import ResultsPlotter


# CLASS


'''
    This class is responsible for running all the different parts of the application.
    It uses an instance of the DataApi for fetching the data from the csv files, and
    then passes that data to the Preprocessor for preprocessing and possibly scrambling.

    Then it performs 10-fold cross validation on the data using the CrossValidation class,
    and runs the algorithms on each combination of training/test sets using the
    algorithms classes. The intermediate loss function results are calculated using the
    Results class, and the run_experiment() method below returns the average loss
    function results using zero one loss and mean squared error as error metrics.

    The entire set of experiments is kicked off by simply running this script.
'''


class ExperimentRunner():


    def __init__(self):
        self.DEBUG = False

        # get instances of all the classes needed to run an experiment
        self.data_api_impl = DataApi('../../data/')
        self.preprocessor_impl = Preprocessor()
        self.cross_validator_impl = CrossValidator()
        self.parameter_tuner_impl = ParameterTuner()

        # algorithm implementations
        self.knn_impl = KNN()
        self.enn_impl = EditedKNN()
        self.cnn_impl = CondensedKNN()
        self.kmeans_knn_impl = KMeansClustering()
        self.k_medoids_clustering_impl = KMedoidsClustering()

        self.results_processor_impl = Results()

        self.CLASSIFICATION = False
        self.REGRESSION = False


    # run algorithm on data set with various parameters
    def run_experiment(self, data_frame_name, algorithm):

        self.set_experiment_type(data_frame_name)

        # get raw data frame to run experiment against
        raw_data_frame = self.data_api_impl.get_raw_data_frame(data_frame_name)
        print(raw_data_frame)

        # preprocess data
        preprocessed_data_frame = self.preprocessor_impl.preprocess_raw_data_frame(raw_data_frame, data_frame_name)
        print(preprocessed_data_frame)

        # get indexes list for data frame cross validation - a list of row numbers used to partition the data
        data_frame_indexes_list = self.cross_validator_impl.get_indexes_list(preprocessed_data_frame)

        if self.DEBUG:
            print('\ndata_frame_name --> ' + data_frame_name)
            print('\nraw_data_frame:\n')
            print(raw_data_frame)
            print('\npreprocessed_data_frame:\n')
            print(preprocessed_data_frame)
            print('\ndata_frame_indexes_list for cross validation:\n')
            print(data_frame_indexes_list)

        # nested dictionary to hold algorithm performance results for each combination of training/test sets
        # key pattern --> key = test_set_1 , where the number at the end of the key is the test set index
        # each value is another dictionary with keys = { 'zero_one_loss', 'mean_squared_error' }
        # the nested dictionary values are the corresponding loss function metrics for predictions using the test set
        cross_validation_results = {}

        # list of sizes of test sets used for getting average test set size
        test_set_sizes = []

        algorithm_parameters = self.parameter_tuner_impl.get_params(data_frame_name, algorithm)
        # dictionary where key is parameter and value is tuple of average loss function results
        results_by_parameter = {}

        # get all cross validation partitions for given data frame
        cv_partitions = self.cross_validator_impl.get_cv_partitions(preprocessed_data_frame)

        # for each parameter value in the list of algorithm parameter values (see ParameterTuner)
        for parameter in algorithm_parameters:

            if self.DEBUG:
                print('\n' + str(self.parameter_tuner_impl.get_parameter_key(algorithm)) + ': ' + str(parameter) + '\n')

            # for each test set used in cross validation (number of folds)
            for partition in cv_partitions:

                # initialize key and corresponding nested dictionary in results dictionary
                test_set_key = 'test_set_' + str(partition)
                cross_validation_results[test_set_key] = {}

                # get training set and test set for given cross validation partition
                training_set, test_set = cv_partitions[partition]

                test_set_sizes.append(test_set.shape[0]) # add number of rows in test set to test_set_sizes list

                if self.DEBUG:
                    print('preprocessed dataframe before running algorithm:')
                    print(preprocessed_data_frame)

                # run algorithms on training set / test set combination
                # returns dictionary where key is the row index (as string) and value is the predicted class for that row
                prediction_results = self.run_algorithm(data_frame_name, algorithm, training_set, test_set, \
                                                            preprocessed_data_frame, parameter)

                # calculate loss function results given prediction results - measure prediction accuracy
                accuracy, mean_squared_error = self.results_processor_impl.loss_function_analysis(test_set, prediction_results)

                cross_validation_results[test_set_key]['accuracy'] = accuracy
                cross_validation_results[test_set_key]['mean_squared_error'] = mean_squared_error

            # calculate average loss function results over all cross validation folds
            avg_accuracy, avg_mean_squared_error = self.results_processor_impl.get_avg_loss_vals(cross_validation_results)
            avg_test_set_size = sum(test_set_sizes) / len(test_set_sizes) # get average test set size for reference

            results_by_parameter[str(parameter)] = (avg_accuracy, avg_mean_squared_error)

            print('\n\nRESULTS: average test set size: ' + str(avg_test_set_size) + \
                ((' --> accuracy: ' + str(avg_accuracy)) if self.CLASSIFICATION \
                else (' --> mean_squared_error: ' + str(avg_mean_squared_error))))

            print('\n---------------------------------------------------------------------------------------------------------------------')

        # return dictionary of results by parameter
        return results_by_parameter


    def set_experiment_type(self, data_frame_name):
        if data_frame_name in ['abalone', 'car', 'segmentation']:
            self.CLASSIFICATION = True
            self.REGRESSION = False
        elif data_frame_name in ['machine', 'forestfires', 'wine']:
            self.REGRESSION = True
            self.CLASSIFICATION = False
        else:
            raise Exception('ERROR: unknown data_set_name --> ' + str(data_frame_name))


    '''
    run algorithm execution handler given algorithm name

    INPUT:
        - algorithm_name: name of algorithm to run handler for

    OUTPUT:
        - prediction results dictionary, maps instance index to tuple: (prediction, actual)
    '''
    def run_algorithm(self, data_set_name, algorithm_name, training_set, \
                        test_set, preprocessed_data_frame, parameter):
        if algorithm == 'knn':
            self.knn_impl.set_data_set(data_set_name)
            self.knn_impl.set_algorithm_name(algorithm_name)
            return self.knn_impl.do_knn(training_set, test_set, preprocessed_data_frame, parameter)
        elif algorithm == 'enn':
            self.enn_impl.set_data_set(data_set_name)
            self.enn_impl.set_algorithm_name(algorithm_name)
            return self.enn_impl.do_enn(training_set, test_set, preprocessed_data_frame, parameter)
        elif algorithm == 'cnn':
            self.cnn_impl.set_data_set(data_set_name)
            self.cnn_impl.set_algorithm_name(algorithm_name)
            return self.cnn_impl.do_cnn(training_set, test_set, preprocessed_data_frame, parameter)
        elif algorithm == 'kmeans_knn':
            self.kmeans_knn_impl.set_data_set(data_set_name)
            self.kmeans_knn_impl.set_algorithm_name(algorithm_name)
            return self.kmeans_knn_impl.cluster_do_knn(training_set, test_set, preprocessed_data_frame, data_set_name, parameter)
        elif algorithm == 'kmedoids_knn':
            self.k_medoids_clustering_impl.set_data_set(data_set_name)
            self.k_medoids_clustering_impl.set_algorithm_name(algorithm_name)
            return self.k_medoids_clustering_impl.cluster(training_set, test_set, preprocessed_data_frame, data_set_name, parameter)



# EXECUTE SCRIPT


# if we run this script directly, the following code below will execute
if __name__ == "__main__":

    print('\nrunning experiments...')

    # get instance of experiment runner class
    experiment_runner_impl = ExperimentRunner()
    #results_plotter_impl = ResultsPlotter()

    # list of data sets to run experiments on
    # possible values: abalone, car, forestfires, machine, segmentation, wine
    data_sets_names = ['segmentation']

    # list of algorithms to run on list of data sets (with various parameters)
    # possible values: knn, enn, cnn, kmeans_knn, kmedoids_knn
    algorithm_names = ['knn']

    # dictionary for storing results - key: data set & algorithm combination
    results_dict = {}

    # run experiment on each data set with each algorithm and various parameters
    for data_set in data_sets_names:
        results_dict[data_set] = {} # nested dictionary for results by algorithm (by data set)
        for algorithm in algorithm_names:
            # save results of experiment to results dictionary - key: data set & algorithm combination
            results_dict[data_set][algorithm] = experiment_runner_impl.run_experiment(data_set, algorithm)

    print('\n\n------------------------------------------------ RESULTS ------------------------------------------------\n')

    # nested dictionary of dictionaries, each containing a tuple: (parameter key, parameter vals, accuracy vals)
    # used for plotting the results of the experiments to show performance impact given change in parameter value
    plotting_results = {}

    # for each data set we have results for
    for data_set in data_sets_names:

        # nested list for each algorithm ran against the data set
        plotting_results[data_set] = {}

        # for each algorithm we ran against the given data set
        for algorithm in algorithm_names:

            # read in experiment results to local variable
            results_by_parameter = results_dict[data_set][algorithm]
            print('\t' + data_set + ':')
            print('\t\t' + algorithm + ':')

            # get parameter key for given algorithm, i.e. 'k' for knn
            parameter_key = experiment_runner_impl.parameter_tuner_impl.get_parameter_key(algorithm)
            # get lists of parameter/accuracy values used for plotting results
            parameter_vals = []
            accuracy_vals = []

            # for each parameter value used
            for parameter in results_by_parameter:
                # accuracy is first index in results tuple
                accuracy = results_by_parameter[parameter][0]
                mean_squared_error = results_by_parameter[parameter][1]

                # lists used for plotting the results
                parameter_vals.append(parameter)
                accuracy_vals.append(accuracy)

                # print to command line for demo
                if data_set in ['abalone', 'car', 'segmentation']:
                    print('\t\t\tACCURACY: ' + str(parameter_key) + ': ' + str(parameter) + ' --> ' + str(accuracy))
                elif data_set in ['machine', 'forestfires', 'wine']:
                    print('\t\t\tMEAN SQUARED ERROR: ' + str(parameter_key) + ': ' + str(parameter) + ' --> ' + str(mean_squared_error))

            # save tuple of results values to plotting_results dictionary used for plotting results
            plotting_results[data_set][algorithm] = (str(parameter_key), parameter_vals, accuracy_vals)


    '''
    try:
        print('\nplotting results...\n')
        # plot results using plotting_results dictionary to show performance impact given change in parameter value
        results_plotter_impl.plot_results(plotting_results)
    except:
        print('ERROR: error plotting results.')
    '''

    print('\n---------------------------------------------------------------------------------------------------------------------')

    print('\nDONE')


    # DONE

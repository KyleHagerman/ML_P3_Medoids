#!/usr/bin/env python3


# IMPORTS


import sys
sys.path.append('../data_api')
sys.path.append('../../../data')

from data_api import DataApi

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import time
import os


# CLASS


'''
    This class handles plotting the results.
'''

class ResultsPlotter:


    def __init__(self):
        self.DEBUG = False
        self.data_api_impl = DataApi('../../data/')


    '''
    TODO: change this so it plots results for all algorithms on the same plot
    
    plot results to show performance impact as algorithm parameter (k) changes

    INPUT:
        - dictionary that maps algorithm name to a tuple containing:
            - the parameter key (k) and two lists of equal length:
                - list 1: parameter values used to get results
                - list 2: prediction accuracy result given parameter value

    OUTPUT:
        - a line plot showing the relationship between parameter value and prediction accuracy
        - it also saves the plot to the /results/plots/ directory for later reference
    '''
    def plot_results(self, results):

        for data_set in results:
            for algorithm in results[data_set]:

                plt.style.use('seaborn-whitegrid')
                fig = plt.figure()
                axes = plt.axes()

                parameter_key = results[data_set][algorithm][0]
                parameter_vals = results[data_set][algorithm][1]
                accuracy_vals = results[data_set][algorithm][2]
                axes.plot(parameter_vals, accuracy_vals)

                plt.xlabel(str(parameter_key))
                plt.ylabel('prediction accuracy')
                plt.title('data: ' + data_set + ', algorithm: ' + algorithm + ' --> parameter vs accuracy')
                plt.show()

                '''
                try:
                    file_name = self.get_file_name_str()
                    print(file_name)
                    fig.savefig(file_name)
                except:
                    print('ERROR: error saving the plot: %s' % file_name)
                '''


    # get .png filename with timestamp
    def get_file_name_str(self):
        filename = 'plots/' + str(time.time()) + '_results_plot'
        return filename + '.png'



# EXECUTE SCRIPT


if __name__ == "__main__":

    print('running results...')

    results_plotter_impl = ResultsPlotter()

    #data_api_impl = DataApi('../../data/')
    #wine_data = data_api_impl.get_raw_data_frame('wine')

    mock_results = {}
    mock_results['wine'] = {}
    mock_results['wine']['knn'] = ('k', [       25,                 100,                500,                1000], \
                                        [0.9906633184018823, 0.9561665761730239, 0.7192236776616443, 0.49346070313560475])

    results_plotter_impl.plot_results(mock_results)


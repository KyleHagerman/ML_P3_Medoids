#!/usr/bin/env python3


# IMPORTS

import numpy as np
import sys
sys.path.append('../data_api')
sys.path.append('../../../data')

from data_api import DataApi


# CLASS


'''
    This class handles processing and saving the results of the runs of the algorithms.
'''

class Results:


    def __init__(self):
        self.DEBUG = False
        self.data_api_impl = DataApi('../../data/')


    # calculate zero/one loss value and mean squared error for given set of predictions
    def loss_function_analysis(self, test_data, predictions_dict):
        count = 0
        right = 0
        wrong = 0
        mse = 0
        for x in predictions_dict:
            count += 1
            predicted_value = predictions_dict[x][0]
            actual_value = predictions_dict[x][1]
            if predicted_value == actual_value:
                right += 1
            else:
                wrong += 1
            if(not isinstance(predicted_value,str)):
                mseval=predicted_value-actual_value
                mseval=np.square(mseval)
                mse=mse+mseval
    
        if count != 0:
            accuracy = right / count
            mse = mse / count
        else:
            accuracy = None
            mse = None

        # return tuple of loss function values (zero_one_loss, mean_squared_error)
        return (accuracy, mse)


    # get average loss function values (zero/one loss and mean squared error) for given cross validation results
    def get_avg_loss_vals(self, cross_validation_results):
        accuracy_vals = []
        mean_squared_error_vals = []
        # for each cross validation partition, append loss function values to corresponding lists
        for test_set_key in cross_validation_results:
            test_set_results = cross_validation_results[test_set_key]
            accuracy_vals.append(test_set_results['accuracy'])
            mean_squared_error_vals.append(test_set_results['mean_squared_error'])

        # should always equal the value of the 'folds' variable in cross validator
        test_set_count = len(cross_validation_results)

        accuracy_vals = self.filter_vals(accuracy_vals)
        mean_squared_error_vals = self.filter_vals(mean_squared_error_vals)

        # calculate average values
        avg_accuracy = sum(accuracy_vals) / test_set_count
        avg_mean_squared_error = sum(mean_squared_error_vals) / test_set_count

        # return tuple with average values for zero_one_loss and mean_squared_error
        return (avg_accuracy, avg_mean_squared_error)


    def filter_vals(self, vals):
        filtered_vals = []
        for val in vals:
            if val is not None:
                filtered_vals.append(val)

        return filtered_vals


# EXECUTE SCRIPT


if __name__ == "__main__":

    print('running results...')

    results_impl = Results()
    data_api_impl = DataApi('../../data/')

    wine_data = data_api_impl.get_raw_data_frame('wine')
    

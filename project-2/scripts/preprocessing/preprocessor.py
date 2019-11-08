#!/usr/bin/env python3


# IMPORTS


import sys
sys.path.append('../data_api')
sys.path.append('../utilities')
sys.path.append('../../../data')

from data_api import DataApi
from utilities import Utilities
import math
import random
import pandas as pd


# CLASS

'''
    This class handles preprocessing all the data sets.
'''


class Preprocessor:


    def __init__(self):
        self.DEBUG = False
        self.data_api_impl = DataApi('../../data/')
        self.utilities_impl = Utilities()


    '''
    main method for this class - return a preprocessed dataframe given a raw dataframe

    INPUT:
        - raw_dataframe: raw dataframe fetched from datalayer
        - data_name: name of data set

    OUTPUT:
        - a preprocessed dataframe ready for consumption by experiment runner
    '''
    def preprocess_raw_data_frame(self, raw_dataframe, data_name):

        prep_dataframe = raw_dataframe.copy()

        # TODO: this can be fixed by changing the datalayer, see the TODO in data_api.py
        # hacky workaround for beginning rows containing labels or weird stuff
        
        if data_name == 'forestfires':
            dtype = 'reg'
        elif data_name == 'machine':
            dtype = 'reg'
        elif data_name == 'wine':
            dtype = 'reg'
        else:
            dtype = 'class'

                    

        # replace or impute missing values depending on ratio of rows with missing values
        prep_dataframe = self.handle_missing_values(prep_dataframe, remove_threshold=0.05)

        # convert categorically-valued columns to numerically-valued columns using value diff metric
        #prep_dataframe = self.handle_categorical_columns(prep_dataframe)

        # normalize dataframe and handle categorical columns
        prep_dataframe = self.normalize(prep_dataframe, dtype)

        # call specific helper methods used for testing - should ultimately be removed
        #prep_dataframe = self.handle_specific(prep_dataframe, data_name)

        # return preprocessed data frame ready for consumption by algorithm classes
        return prep_dataframe


    '''
    handle missing values
        - if ratio of rows with missing values is less than threshold, then just remove rows with missing values
        - otherwise, replace missing values with random values in column range, use column bounds utility method

    INPUT:
        - data frame possibly containing missing values

    OUTPUT:
        - data frame with no missing values, all missing values either removed or imputed
    '''
    def handle_missing_values(self, data_frame, remove_threshold):
        # none of the data sets have missing values so we don't need to do this...
        return data_frame


    #Normalization handler ----- Maps each value to the closed interval [0,1]
    def normalize(self, df, datatype):
        if(datatype == 'class'):
            attributes = list(df.drop("CLASS", axis=1))
        else:
            attributes = list(df)
        #Check what type of data is in each attribute (string or number)
        for val in attributes: #Just check 1st element in each col.
            #print(val)
            if not self.utilities_impl.is_number(df[val][df[val].count()-2]):
                attributemap = self.disc_attribute_map(df[val]) 
            else:
                attributemap = self.cont_attribute_map(df[val])

            df[val]=attributemap
            
        return df


    #Data Conversion Map Generation
    def cont_attribute_map(self, attribute):
        small, big = self.utilities_impl.get_min_max(attribute)
        #print('Continuous Data')
        attribute_vals = []
        # map each cont value to [0,1]
        for x, val in enumerate(attribute):
            divisor = 1
            if big-small != 0:
                divisor = big-small
            if self.utilities_impl.is_number(val):
                attribute_vals.append((float(val)-small)/divisor)
            else:
                attribute_vals.append(None) # workaround for now

        return attribute_vals


    def disc_attribute_map(self, attribute):
        #given an attribute, associate the number of unique values in a list


        uniquevals = attribute.unique()
        #Map each attribute value to a number (E.G. low med high maps to 0 1 2)
        #Then scales to a 0-1 value (For comparison)
        for i, valuetype in enumerate(uniquevals):
            uniquevals[i] = i/(len(uniquevals)-1)
        normalized = attribute.replace(attribute.unique(),uniquevals)

        return normalized


# EXECUTE SCRIPT


if __name__ == '__main__':

    print('running preprocessor...')
    preprocessor_impl = Preprocessor()

    data_api_impl = DataApi('../../data/')

    '''
    raw_abalone_data = data_api_impl.get_raw_data_frame('abalone')
    print('raw_abalone_data:')
    print(raw_abalone_data)
    prep_abalone_data = preprocessor_impl.preprocess_raw_data_frame(raw_abalone_data, 'abalone')
    print('prep_abalone_data:')
    print(prep_abalone_data)
    '''

    '''
    raw_car_data = data_api_impl.get_raw_data_frame('car')
    print('raw_car_data:')
    print(raw_car_data)
    prep_car_data = preprocessor_impl.preprocess_raw_data_frame(raw_car_data, 'car')
    print('prep_car_data:')
    print(prep_car_data)
    '''

    '''
    raw_ff_data = data_api_impl.get_raw_data_frame('forestfires')
    print('raw_ff_data:')
    print(raw_ff_data)
    prep_ff_data = preprocessor_impl.preprocess_raw_data_frame(raw_ff_data, 'forestfires')
    print('prep_ff_data:')
    print(prep_ff_data)
    '''

    '''
    raw_machine_data = data_api_impl.get_raw_data_frame('machine')
    print('raw_machine_data:')
    print(raw_machine_data)
    prep_machine_data = preprocessor_impl.preprocess_raw_data_frame(raw_machine_data, 'machine')
    print('prep_machine_data:')
    print(prep_machine_data)
    '''

    '''
    raw_segmentation_data = data_api_impl.get_raw_data_frame('segmentation')
    print('raw_segmentation_data:')
    print(raw_segmentation_data)
    prep_segmentation_data = preprocessor_impl.preprocess_raw_data_frame(raw_segmentation_data, 'segmentation')
    print('prep_segmentation_data:')
    print(prep_segmentation_data)
    '''

    '''
    raw_wine_data = data_api_impl.get_raw_data_frame('wine')
    print(raw_wine_data)

    normalized_wine_data = preprocessor_impl.normalize(raw_wine_data)
    print(normalized_wine_data)
    '''
    #Check all preprocessed data sets
    datasets = ['abalone', 'car', 'segmentation', 'machine', 'forestfires', 'wine']
    for data in datasets:
        rawdata = data_api_impl.get_raw_data_frame(data)
        print(str(data))
        print(rawdata)
        print('Preprocessed: ' + str(data))
        print(preprocessor_impl.preprocess_raw_data_frame(rawdata,data))
    '''

    raw_car_data = data_api_impl.get_raw_data_frame('car')
    print('raw_car_data:')
    print(raw_car_data)

    normalized_car_data = preprocessor_impl.normalize(raw_car_data)
    print(normalized_car_data)
    '''

    #prep_wine_data = preprocessor_impl.preprocess_raw_data_frame(raw_wine_data, 'wine')
    #print(prep_wine_data.shape)


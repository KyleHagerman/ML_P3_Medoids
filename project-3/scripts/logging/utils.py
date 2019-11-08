#!/usr/bin/env python3


# IMPORTS


import sys


# CLASS

'''
    Utilities stuff
'''

class Utils:


    '''
    CONSTRUCTOR

    args:
        
    '''
    def __init__(self):
        self.DEBUG = False


    # get list of shapes of matrices in arg
    def get_shapes(self, matrices_list):
        shapes = []
        for matrix in matrices_list:
            shapes.append(matrix.shape)
        return shapes

    '''
    get booleans for determining whether we're doing classification or regression

    INPUT:
        - data_set: name of data set

    OUTPUT:
        - return tuple of booleans: (CLASSIFICATION, REGRESSION)
    '''
    def get_classification_regression_vals(self, data_set):
        if data_set in ['abalone', 'car', 'segmentation']:
            # CLASSIFICATION data sets
            return (True, False)
        elif data_set in ['machine', 'forestfires', 'wine']:
            # REGRESSION data set
            return (False, True)
        else:
            raise Exception('Invalid data set name: %s' % str(data_set))



# EXECUTE SCRIPT


if __name__ == '__main__':

    print('running utils...')


#!/usr/bin/env python3


# IMPORTS


import sys
# add following directories to class path
sys.path.append('../logging')

from neural_network import NeuralNetwork
from logger import Logger


# CLASS

'''
    This class handles all things multilayer perceptron network.
    It inherits from the NeuralNetwork base class, and does not need to override anything.
'''

class MLPNetwork(NeuralNetwork):


    '''
    CONSTRUCTOR

    args:
        - layer_sizes: list of layer sizes, size of list is number of layers in network
            - for example: [10, 5, 2] would be the number of nodes for the input layer (10), 
                            the single hidden layer (5), and the output layer (2) respectively
    '''
    def __init__(self, data_set_name, layer_sizes):
        NeuralNetwork.__init__(self, data_set_name, layer_sizes)

        self.logger = Logger('DEMO') # configure class-level log level here

        self.network_name = 'MLP'



# EXECUTE SCRIPT


if __name__ == '__main__':

    print('\nrunning MLPNetwork...\n')



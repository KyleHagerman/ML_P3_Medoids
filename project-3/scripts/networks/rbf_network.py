#!/usr/bin/env python3


# IMPORTS


import sys
sys.path.append('../../../project-2/scripts/algorithms')
sys.path.append('../../data')
sys.path.append('../networks')
sys.path.append('../logging')
sys.path.append('../../../project-2/scripts/data_api')
sys.path.append('../../../project-2/scripts/preprocessing')
sys.path.append('../../../project-2/scripts/cross_validator')
sys.path.append('../../../project-2/scripts/utilities')
sys.path.append('../../../project-2/scripts/algorithms')

from neural_network import NeuralNetwork
from project2_API import P2API
from logger import Logger
from data_api import DataApi
from scipy.spatial import distance
import numpy as np


# CLASS


'''
This class handles all things radial basis function network.
It implements the NeuralNetwork base class.

To train:
    - define (beta, mu) coefficients for all RBF neurons in middle layer
    - can experiment with random beta values I think
    - mu is just going to be the training point or centroid/medoid from the P2 algorithms
    - learn matrix of weights between middle layer (RBF neurons) and output layer
'''


class RBFNetwork(NeuralNetwork):


    '''
    CONSTRUCTOR

    args:
        - layer_sizes: list of layer sizes, size of list is number of layers in network
        - for example: [10, 5, 2] would be the number of nodes for the input layer (10),
                            the RBF neuron layer (5), and the output layer (2) respectively
    '''
    def __init__(self, data_set_name, layer_sizes, P2_algorithm_basis=None):
        NeuralNetwork.__init__(self, data_set_name, layer_sizes)

        self.network_name = 'RBF'

        self.logger = Logger('DEMO')

        self.P2_algorithm_basis = P2_algorithm_basis
        # instance of Project 2 API - a single class for accessing all things needed from P2
        self.P2API = P2API(data_set_name, self.P2_algorithm_basis)

        # list of tuples for RBF neuron values, each tuple is (beta, mu)
        # beta at spot 1 and vector is at spot 0
        self.rbf_neurons = []


    '''
    reconfigure network shape based on P2 results using training data

    INPUT:
        - train_data: P3 training data to run P2 algorithms on to get RBF neuron data
        - full_data: full data frame
        - data_set_name: name of data set we're working with
        - k: value of k parameter used in all algorithms from P2

    OUTPUT:
        - <void> just reconfigure the network appropriately
    '''
    def configure_rbf_network(self, train_data, full_data, data_set_name, k):
        self.logger.log('DEMO', 'resetting RBF layer...')
        self.rbf_neurons = [] # reset RBF layer for next cross validation partition using P2 results

        self.logger.log('DEMO', 'configuring RBF network with P2 algorithm basis: \t%s' % str(self.P2_algorithm_basis))

        if self.P2_algorithm_basis == 'enn':
            edited_train_data = self.P2API.get_enn_for_rbf_network(train_data, full_data, k)
            self.layer_sizes[1] = edited_train_data.shape[0]
            self.init_RBF_Neurons(edited_train_data)
            self.logger.log('DEMO', 'edited_train_data.shape: %s\n' % str(edited_train_data.shape))

        elif self.P2_algorithm_basis == 'cnn':
            condensed_train_data = self.P2API.get_cnn_for_rbf_network(train_data, full_data, k)
            self.layer_sizes[1] = condensed_train_data.shape[0]
            self.init_RBF_Neurons(condensed_train_data)
            self.logger.log('DEMO', 'condensed_train_data.shape: %s\n' % str(condensed_train_data.shape))

        elif self.P2_algorithm_basis == 'kmeans_knn':
            kmeans_centroids_data = self.P2API.get_kmeans_centroids_for_rbf_network(train_data, full_data, k)
            self.layer_sizes[1] = kmeans_centroids_data.shape[0]
            self.init_RBF_Neurons(kmeans_centroids_data)
            self.logger.log('DEMO', 'kmeans_centroids_data.shape: %s\n' % str(kmeans_centroids_data.shape))

        elif self.P2_algorithm_basis == 'kmedoids_knn':
            kmedoids_medoids_data = self.P2API.get_kmedoids_medoids_for_rbf_network(train_data, full_data, k)
            self.layer_sizes[1] = kmedoids_medoids_data.shape[0]
            self.init_RBF_Neurons(kmedoids_medoids_data)
            self.logger.log('DEMO', 'kmedoids_medoids_data.shape: %s\n' % str(kmedoids_medoids_data.shape))


    # OVERRIDE: do forward propagation for RBF network
    def forward_prop(self, train_data_vector):
        #get rbf activation calculation for every RBF node within the network.
        rbf_output_list = [self.calculate_RBF_activation(train_data_vector, self.rbf_neurons[x][0], self.rbf_neurons[x][1]) for x in range(len(self.rbf_neurons))]
        self.logger.log('DEBUG', 'rbf_output_list: %s, len: %s' % (str(rbf_output_list), str(len(rbf_output_list))))

        #get out_put calculation for every RBF node
        output_list, activations = self.calculate_output(rbf_output_list)
        self.logger.log('DEBUG', 'output_list: %s, len: %s' % (str(output_list), str(len(activations))))
        return (output_list, activations)

        #---------------------------------------------------------------------------------------
        #Tried to reproduce ^this with list comprehension below
        #values = [[self.calculate_RBF(z, self.rbf_neurons[y][0], self.rbf_neurons[y][1]) \
        #               for y in range(len(self.rbf_neurons))] for z in range(len(test_vector))]
        #---------------------------------------------------------------------------------------


    #input: activated RBF neurons
    #output: (1) activation_list - the outpute activated neurons (2) output_list is the list of all values summed for each output neuron. Will be used in training
    def calculate_output(self, RBF_list):
        biased_list = [float(x) for x in self.biases[0]]
        #output list will be alist
        output_list = []

        for x in range(len(self.biases[0])): #because each biases correpsondes to one out put node
            weight_vector = self.weights[0][x,:]

            self.logger.log('DEBUG', 'weight_vector: %s, shape: %s' % (str(weight_vector), str(weight_vector.shape)))
            RBF_list = np.array(RBF_list).reshape(len(RBF_list), 1)
            biased_list = np.array(biased_list).reshape(len(biased_list), 1)
            self.logger.log('DEBUG', 'RBF_list: %s, shape: %s' % (str(RBF_list), str(RBF_list.shape)))
            self.logger.log('DEBUG', 'biased_list: %s, shape: %s' % (str(biased_list), str(biased_list.shape)))
            # output_list corresponds to is a list of lists, where each elementer[x][y] is each RBF's output for 1 output node, to be used in back prop
            output_value = np.dot(weight_vector, RBF_list) + biased_list[x]

            self.logger.log('DEBUG', 'output_value_list: %s' % str(output_value))
            output_list.append(output_value)

        output_list = [float(x) for x in output_list]
        activations = [self.sigmoid(x) for x in output_list]

        # convert lists to numpy column vectors for compatibility with base class methods
        output_list = np.array(output_list).reshape(len(output_list), 1)
        activations = np.array(activations).reshape(len(activations), 1)
        RBF_list = np.array(RBF_list).reshape(len(RBF_list), 1)

        # list of activation vectors for RBF layer (the RBF function values) and output layer
        layer_activations = [RBF_list, activations]

        # return tuple with list of z vectors and list of layer activation vectors
        return ([output_list], layer_activations)


    # initialize RBF neurons given dataframe from P2 results
    def init_RBF_Neurons(self, RBF_dataframe):
        #------(1)find average vector point given RBF_dataframe-------
        RBF_dataframe = RBF_dataframe.loc[:, RBF_dataframe.columns != 'CLASS']
        avg_vector = []
        value = 0
        for y in range(RBF_dataframe.shape[1]):
            for x in range(RBF_dataframe.shape[0]):
                value += RBF_dataframe.iloc[x][y]
            avg_vector.append(value)
            value = 0
        avg_vector = [x / RBF_dataframe.shape[0] for x in avg_vector]
        #-------------------------Get RBF value, and beta value and store in self.rbf_neurons----------------------------
        our_list = []
        for y in range(RBF_dataframe.shape[0]):
            row_vector = RBF_dataframe.iloc[y,0:].ravel()
            beta = (1/(2*abs(distance.euclidean(row_vector, avg_vector))**2))
            our_list.append(row_vector)
            our_list.append(beta)
            self.rbf_neurons.append(our_list)
            our_list = []


    # calculate activation value for RBF neuron based on RBF function for particular node
    def calculate_RBF_activation(self, input_vector, RBF_vector, RBF_beta):
        return np.exp(-1 * RBF_beta * ((distance.euclidean(input_vector, RBF_vector)**2)))


    # OVERRIDE: initialize weights and biases for RBF network
    def init_weights_biases(self):
        self.weights = [np.random.randn(self.layer_sizes[2], self.layer_sizes[1])]
        # the first node in every list correpsonds to the the 1 output from top to bottom. Print-wise, this means each row
        #,as it is printed, correpsonds to the to all weights cominging in to output node i.
        #ensure that we only have layers 1 and 2 b/c RBF will always be of size 3
        self.biases = [np.random.randn(self.layer_sizes[2], 1)]



# EXECUTE SCRIPT

if __name__ == '__main__':

    print('\nrunning RBFNetwork...\n')
    data_set_name = 'segmentation'
    data_api_impl = DataApi('../../data/')
    data = data_api_impl.get_raw_data_frame(data_set_name)
    rbf_network_impl = RBFNetwork(data_set_name, [19, 2, 7])
    #initializing neuron values
    rbf_network_impl.init_RBF_Neurons(data[0:2])
    #print(rbf_network_impl.rbf_neurons)
    rbf_network_impl.init_weights_biases()

    print('weights: %s' % str(rbf_network_impl.weights))
    print('biases: %s' % str(rbf_network_impl.biases))
    #test_vector = [121.0,60.0,9,0.0,0.0,2.277778,2.329629,2.888889,2.8740742,26.74074,24.666666,35.22222,20.333334,-6.2222223,25.444445,-19.222221,35.22222,0.4223002,-1.776113]
    #rbf_network_impl.get_forward_output(test_vector)

    #print(rbf_network_impl.rbf_neurons)
    #print(data.loc[0,:])
    #print(rbf_network_impl.calculate_std(data))

   # print(rbf_network_impl.weights)
    #print(rbf_network_impl.weights[:,0])

    #print(rbf_network_impl.biases)

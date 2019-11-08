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
from logger import Logger
from utils import Utils

import numpy as np


# CLASS

'''
    This class handles all things neural network. Base class for MLPNetwork and RBFNetwork classes.

    CONVENTIONS:

    CLASSIFICATION:
        - the output layer consists of n nodes, one for each possible class value
            - the output layer nodes will be in order specified in get_ordered_output_nodes() below

    REGRESSION:
        - the output layer always consists of a single node (the regression output in range [0,1])
'''

class NeuralNetwork:


    '''
    CONSTRUCTOR

    args:
        - data_set_name: name of data set network will operate with
        - layer_sizes: list of layer sizes, size of list is number of layers in network
            - for example: [10, 5, 2] would be the number of nodes for the input layer (10), 
                            the single hidden layer (5), and the output layer (2) respectively
    '''
    def __init__(self, data_set_name, layer_sizes):
        # logger instance - VERBOSE level is highest (most verbose) level for logging
        self.logger = Logger('DEMO')
        # utils instance - random things
        self.utils = Utils()

        # name of data set network will train/test with
        self.data_set_name = data_set_name

        # set booleans for determining whether we're doing CLASSIFICATION or REGRESSION
        self.CLASSIFICATION, self.REGRESSION = self.utils.get_classification_regression_vals(self.data_set_name)
        self.logger.log('DEMO', 'CLASSIFICATION: %s, REGRESSION: %s\n' % (self.CLASSIFICATION, self.REGRESSION))

        # validate layer sizes, verify i/o shape is valid for data set
        #assert self.validate_layer_sizes(self.data_set_name, layer_sizes)
        # list of layer sizes that define network shape
        self.layer_sizes = layer_sizes
        # list to specify which activation function to use for each layer
        self.layer_activation_funcs = None

        # list of matrices of weights for all valid pairs of layers in network
        self.weights = None
        # list of bias vectors for biases in each valid layer in network
        self.biases = None
        # list of previous nabula gradients for weights - used in momentum calculation
        self.prev_nab_weights = None
        # list of previous nabula gradients for biases - used in momentum calculation
        self.prev_nab_biases = None

        # learning rate eta
        self.eta = None


    '''
    train network with gradient descent using all batches of training data

    INPUT:
        - train_data: training data to train network on
        - hyperparams:
            - (possible values include)
                - max_iterations: maximum number of iterations for gradient descent
                - batch_size: size of batches to use in mini-batch training
                - eta: learning rate
                - layer_activation_funcs: list of activation functions to use for each layer
                - use_momentum: boolean indicating whether to use momentum when training
                - momentum_beta: beta param value for momentum, in range [0,1]
        - cv_partition: cross validation partition number
        - (optional) test_data: test data frame

    OUTPUT:
        - return list of result values (accuracy/error) achieved after each iteration of gradient descent
    '''
    def train_gradient_descent(self, train_data, hyperparams, cv_partition, test_data=None):

        # activation function specified by layer, options: ['sigmoid', 'relu', 'tanh']
        self.layer_activation_funcs = hyperparams["layer_activation_funcs"]

        # randomly initialize all weights/biases lists in network
        self.init_weights_biases()

        # get values for hyperparams used in this method
        max_iterations = hyperparams["max_iterations"]
        batch_size = hyperparams["batch_size"]

        result_vals = []

        # for each iteration in gradient descent
        for iteration in range(max_iterations):
            # construct list of mini batches for current iteration 
            # randomly shuffle data so the batches are different every iteration
            shuffled_train_data = train_data.sample(frac=1)
            batches = [shuffled_train_data[batch_idx:batch_idx+batch_size] \
                        for batch_idx in range(0, shuffled_train_data.shape[0], batch_size)]

            self.logger.log('INFO', 'num mini batches: %s' % str(len(batches)))

            for batch in batches:
                self.logger.log('INFO', 'batch.shape: ' + str(batch.shape))
                # run backprop using given batch and update weights/biases after batch is complete
                self.do_batch(batch, hyperparams)

            if test_data is not None:
                # if test data is specified, run forward propagation using test data
                result_vals.append(self.do_test_data(cv_partition, iteration, test_data))

        # return list of result values (accuracy/error) achieved for each iteration in gradient descent
        return result_vals


    '''
    randomly initialize all weights and biases before first iteration of gradient descent
        - zero initialize previous nabula gradients for weights/biases - used in momentum calculation
    '''
    def init_weights_biases(self):
        # randomly initialize list of matrices of weights for all valid pairs of layers in network
        self.weights = [np.random.randn(next_l_size, prev_l_size) \
                        for prev_l_size, next_l_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]

        # randomly initialize list of bias vectors for biases in each valid layer in network
        self.biases = [np.random.randn(l_size, 1) for l_size in self.layer_sizes[1:]]

        # zero initialize previous nabula gradient for weights - used in momentum calculation
        self.prev_nab_weights = [np.zeros(w_matrix.shape) for w_matrix in self.weights]
        # zero initialize previous nabula gradient for biases - used in momentum calculation
        self.prev_nab_biases = [np.zeros(b_vector.shape) for b_vector in self.biases]


    '''
    update weights and biases after calculating average gradient for mini batch

    INPUT:
        - batch: batch to use for training and updating weights/biases
        - hyperparams:

    OUTPUT:
        - <void> - this method just updates the class-level weights and biases after doing a single batch
    '''
    def do_batch(self, batch, hyperparams):
        # initialize lists for storing cumulative gradient nudges updated after each backprop
        nab_grad_w = [np.zeros(w_matrix.shape) for w_matrix in self.weights]
        nab_grad_b = [np.zeros(b_vector.shape) for b_vector in self.biases]

        # for each data point in the batch
        for idx, _ in batch.iterrows():
            # NOTE: for some reason specifying the data type (dtype) to np.float is required here...
            feature_vector = np.array(batch.loc[idx, batch.columns != 'CLASS'], dtype=np.float)
            # reshape feature vector into column vector with length equal to number of attributes
            # NOTE: subtract 1 from number of columns because feature vector does not have CLASS column
            feature_vector = feature_vector.reshape(batch.shape[1]-1, 1)
            # get vector representing expected output - i.e. [0, 0, 0, 0, 1, 0, 0, 0]
            expected_output = self.get_expected_output(batch.loc[idx, 'CLASS'])

            # do backpropagation to get the gradient nudges for a single training point
            del_grad_w, del_grad_b = self.backward_prop(feature_vector, expected_output)
            # add the single-point gradient nudges to the cumulative gradient nudges list
            nab_grad_w = [gw_val + del_gw for gw_val, del_gw in zip(nab_grad_w, del_grad_w)]
            nab_grad_b = [gb_val + del_gb for gb_val, del_gb in zip(nab_grad_b, del_grad_b)]

        # after accumulating the gradient nudges for all points in the batch, update the weights/biases

        self.logger.log('DEBUG', 'nab_grad_w: %s, shape: %s' \
            % (str(nab_grad_w), str(self.utils.get_shapes(nab_grad_w))))
        self.logger.log('DEBUG', 'nab_grad_b: %s, shape: %s' \
            % (str(nab_grad_b), str(self.utils.get_shapes(nab_grad_b))))

        # update weights and biases - use momentum if specified to do so
        self.update_weights_biases(nab_grad_w, nab_grad_b, hyperparams)


    '''
    update class-level weights and biases - use momentum if specified to do so

    INPUT:
        - nab_grad_w: list of gradients for weight matrices
        - nab_grad_b: list of gradients for bias vectors
        - hyperparams:
            - (actual ones used in this method)
                - eta: learning rate
                - use_momentum: boolean indicating whether to use momentum when training
                - momentum_beta: beta param value for momentum calculation

    OUTPUT:
        - <void> - this method just updates the class-level weights and biases
    '''
    def update_weights_biases(self, nab_grad_w, nab_grad_b, hyperparams):
        eta = hyperparams["eta"]
        batch_size = hyperparams["batch_size"]
        momentum_beta = hyperparams["momentum_beta"]
        if hyperparams["use_momentum"]:
            # update weights/biases - use momentum in calculation
            # w_vals[0] is grad_w (the current weight gradient), w_vals[1] is nab_grad_w (the nabula weight gradient)
            # reuse previous nabula weight gradient and save current as previous for reuse in next iteration
            self.prev_nab_weights = [(momentum_beta * self.prev_nab_weights[w_idx]) + ((1-momentum_beta) * w_vals[1]) \
                                                    for w_idx, w_vals in enumerate(zip(self.weights, nab_grad_w))]

            self.weights = [weights - ((eta/batch_size) * nab_weight) \
                                for weights, nab_weight in zip(self.weights, self.prev_nab_weights)]

            self.prev_nab_biases = [(momentum_beta * self.prev_nab_biases[b_idx]) + ((1-momentum_beta) * b_vals[1]) \
                                                    for b_idx, b_vals in enumerate(zip(self.biases, nab_grad_b))]

            self.biases = [biases - ((eta/batch_size) * nab_bias) \
                                for biases, nab_bias in zip(self.biases, self.prev_nab_biases)]

        else:
            # update weights/biases - do NOT use momentum in calculation
            self.weights = [weight - ((eta/batch_size) * nab_weight) \
                            for weight, nab_weight in zip(self.weights, nab_grad_w)]

            self.biases = [bias - ((eta/batch_size) * nab_bias) \
                            for bias, nab_bias in zip(self.biases, nab_grad_b)]


    '''
    calculate all activations using feedforward algorithm given input vector
        - keep track of intermediate z vals to be used in backpropagation

    INPUT:
        - input_vector: input vector we want to feed into the network

    OUTPUT:
        - return tuple of (list of z vectors, list of activation vectors)
    '''
    def forward_prop(self, input_vector):
        self.logger.log('DEBUG', 'forward_prop input_vector: %s, shape: %s\n' % (str(input_vector), str(input_vector.shape)))
        z_vectors = []
        a_vectors = []
        # set input vector as first layer activations, append to list of activations vectors
        activations = input_vector
        a_vectors.append(activations)

        # for each layer index, calculate activations given biases and weights for that layer
        for layer_idx, wb_pair in enumerate(zip(self.biases, self.weights)):

            # get biases and weights from zip output
            biases, weights = wb_pair

            self.logger.log('DEBUG', 'weights: %s, shape: %s\n' % (str(weights), str(weights.shape)))
            self.logger.log('DEBUG', 'biases: %s, shape: %s\n' % (str(biases), str(biases.shape)))
            self.logger.log('DEBUG', 'activations: %s, shape: %s\n' % (str(activations), str(activations.shape)))

            # calculate vector of z values for given layer
            z_vec = np.dot(weights, activations) + biases
            z_vectors.append(z_vec)

            self.logger.log('DEBUG', 'z_vec: %s, shape: %s\n' % (str(z_vec), str(z_vec.shape)))

            # input to squish is weighted sum with bias --> (w (dot) a) + b
            # layer index determines which activation function to use
            activations = self.squish(z_vec, layer_idx, derivative=False)
            a_vectors.append(activations)

        # return tuple of (list of z vectors, list of activation vectors)
        return (z_vectors, a_vectors)


    '''
    compute cost of output activations given expected output

    INPUT:
        - network_output: network output - in form [0.25, 0.67, 0.1, 0.8, 0.05].transpose()
        - expected_output: expected output vector - must be in form [0, 0, 0, 1, 0].transpose()

    OUTPUT:
        - difference vector of diff between network and expected output
    '''
    def compute_cost(self, network_output, expected_output):
        self.logger.log('DEBUG', 'network_output: %s, \nexpected_output: %s' \
                                % (str(network_output), str(expected_output)))
        # the cost is simply the diff between the network output and expected output
        cost = network_output - expected_output
        self.logger.log('DEBUG', 'compute_cost: cost: %s, shape: %s' % (str(cost), str(cost.shape)))
        return cost


    '''
    get column vector representing expected output - i.e. [0, 0, 0, 0, 1, 0, 0, 0].transpose()

    INPUT:
        - output_val: actual output value - i.e. 'PATH' for segmentation data

    OUTPUT:
        - return column vector representing expected output
    '''
    def get_expected_output(self, output_val):
        if self.REGRESSION:
            # for regression there is always a single output node
            return np.array([output_val]).reshape(1, 1)
        else:
            # CLASSIFICATION
            ordered_output_nodes = self.get_ordered_output_nodes()
            # return column vector representing expected output containing only 0/1 values
            return np.array([int(val == output_val) for val in ordered_output_nodes])\
                                                .reshape(len(ordered_output_nodes), 1)


    '''
    calculate gradients for all weights/biases using backpropagation algorithm

    INPUT:
        - input_vector: input vector (training point)
        - expected_output: expected output vector - must be in form [0, 0, 0, 1, 0].transpose()

    OUTPUT:
        - tuple of gradient lists for weights/biases
    '''
    def backward_prop(self, input_vector, expected_output):
        self.logger.log('DEBUG', '\n\nbackward_prop:\n')
        # initialize lists for storing layer-wise gradient nudges for weights/biases
        nab_grad_w = [np.zeros(w_matrix.shape) for w_matrix in self.weights]
        nab_grad_b = [np.zeros(b_vector.shape) for b_vector in self.biases]

        self.logger.log('DEBUG', 'nab_grad_w at start of backward_prop: %s, shape: %s' \
                                            % (str(nab_grad_w), str('SKIP')))
        self.logger.log('DEBUG', 'nab_grad_b at start of backward_prop: %s, shape: %s' \
                                            % (str(nab_grad_b), str('SKIP')))

        # do forward propagation to calculate intermediate z values and activations
        z_vectors, activations = self.forward_prop(input_vector)

        self.logger.log('DEBUG', 'z_vectors after forward_prop: %s, shape: %s' \
                                        % (str(z_vectors), str('SKIP')))
        self.logger.log('DEBUG', 'activations after forward_prop: %s, shape: %s' \
                                        % (str(activations), str('SKIP')))

        # do backward propagation to calculate layer-wise gradient nudges
        diff = self.compute_cost(activations[-1], expected_output) \
                * self.squish(z_vectors[-1], layer_idx = -1, derivative = True)
        
        self.logger.log('DEBUG', 'diff before dot: %s, shape: %s' % (str(diff), str(diff.shape)))
        self.logger.log('DEBUG', 'activations[-2] before dot: %s, shape: %s' % (str(activations[-2]), str('SKIP')))
        
        # update bias and weight gradients for final layer (first considered in backprop)
        nab_grad_b[-1] = diff
        nab_grad_w[-1] = np.dot(diff, activations[-2].transpose())

        self.logger.log('DEBUG', 'nab_grad_b (diff) after compute_cost: %s, shape: %s' \
                                                % (str(nab_grad_b), str('SKIP')))
        self.logger.log('DEBUG', 'nab_grad_w after compute_cost: %s, shape: %s' \
                                                % (str(nab_grad_w), str('SKIP')))

        # stop backprop early for RBF network - only have one weight matrix & bias vector to update
        if self.network_name == 'RBF':
            return (nab_grad_w, nab_grad_b)

        # iterate backwards through remaining layers for MLP network backprop
        for layer_idx in range(2, len(self.layer_sizes)):
            z_vec = z_vectors[-layer_idx]
            # calculate derivative of activation value of z vector
            d_squish = self.squish(z_vec, -layer_idx, derivative=True)
            # calculate next diff using previous diff and sigmoid derivative
            diff = np.dot(self.weights[-layer_idx+1].transpose(), diff) * d_squish
            self.logger.log('DEBUG', 'nab_grad_b: %s, shape: %s' % (str(nab_grad_b), str('SKIP')))
            self.logger.log('DEBUG', 'layer_idx: %s, diff: %s' % (str(layer_idx), str(diff)))
            # update bias and weight gradients corresponding to current layer in loop iteration
            nab_grad_b[-layer_idx] = diff
            nab_grad_w[-layer_idx] = np.dot(diff, activations[-layer_idx-1].transpose())

        # return gradients for all weights/biases across all layers
        return (nab_grad_w, nab_grad_b)


    '''
    make predictions for test_data points and log accuracy results at DEMO level

    INPUT:
        - cv_partition: cross validation partition number
        - iteration: iteration count in gradient descent for given cv partition
        - test_data: test data frame containing all points to predict (for cv partition)

    OUTPUT:
        - return average accuracy/error result - log at DEMO level
    '''
    def do_test_data(self, cv_partition, iteration, test_data):
        # list of values 0/1 where value is 1 if prediction was correct
        test_results = []
        # for each data point in the test_data dataframe
        for idx, _ in test_data.iterrows():
            # get vector of attributes for all columns except the CLASS column, cast as np.float type
            feature_vector = np.array(test_data.loc[idx, test_data.columns != 'CLASS'], dtype=np.float)

            self.logger.log('VERBOSE', 'test feature vector: %s, shape: %s\n' \
                % (str(feature_vector), str(feature_vector.shape)))

            # reshape feature vector into column vector with length equal to number of attributes
            feature_vector = feature_vector.reshape(test_data.shape[1]-1, 1)

            self.logger.log('VERBOSE', 'test feature column: %s, shape: %s\n' \
                % (str(feature_vector), str(feature_vector.shape)))

            # get actual class/regression value from test data frame
            actual = test_data.loc[idx, 'CLASS']

            assert actual is not None and actual is not ''
            self.logger.log('VERBOSE', 'actual class/regression value: %s\n' % str(actual))

            # do forward propagation to calculate activations
            _, activations = self.forward_prop(feature_vector)
            # get network output val - the actual class/regression value prediction
            output = self.get_output_val(activations[-1])
            # append test result to list of test results
            test_results.append(self.handle_prediction(idx, output, actual))

        '''
        get accuracy/error for test data
            - CLASSIFICATION: average accuracy: ratio of correct predictions to total number of predictions
            - REGRESSION: average error: average diff between prediction and actual
        '''
        avg_test_result = round(sum(test_results) / len(test_results), 5) # round to 5 decimals

        if self.CLASSIFICATION:
            self.logger.log('DEMO', 'cv_partition: %s, iteration: %s, accuracy: %s' \
                % (str(cv_partition+1), str(iteration+1), str(avg_test_result)))
        elif self.REGRESSION:
            self.logger.log('DEMO', 'cv_partition: %s, iteration: %s, error: %s' \
                % (str(cv_partition+1), str(iteration+1), str(avg_test_result)))

        return avg_test_result


    # HELPER METHODS


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
    def validate_layer_sizes(self, data_set_name, layer_sizes):
        if data_set_name == 'segmentation':
            return layer_sizes[0] == 19 and layer_sizes[-1] == 7
        elif data_set_name == 'car':
            return layer_sizes[0] == 6 and layer_sizes[-1] == 4
        elif data_set_name == 'abalone':
            return layer_sizes[0] == 8 and layer_sizes[-1] == 28
        elif data_set_name == 'machine':
            return layer_sizes[0] == 9 and layer_sizes[-1] == 1
        elif data_set_name == 'forestfires':
            return layer_sizes[0] == 12 and layer_sizes[-1] == 1
        elif data_set_name == 'wine':
            return layer_sizes[0] == 11 and layer_sizes[-1] == 1
        else:
            raise Exception('Unknown/unhandled data set name: %s' % str(self.data_set_name))


    '''
    IMPORTANT: specify order of output nodes by data set

    NOTE: for regression data sets we only have one output node for regression value
    '''
    def get_ordered_output_nodes(self):
        if self.data_set_name == 'segmentation':
            return ['BRICKFACE', 'CEMENT', 'FOLIAGE', 'GRASS', 'PATH', 'SKY', 'WINDOW']
        elif self.data_set_name == 'car':
            return ['unacc', 'acc', 'good', 'v-good']
        elif self.data_set_name == 'abalone':
            return ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', \
                    '11', '12', '13', '14', '15', '16', '17', '18', '19', \
                    '20', '21', '22', '23', '24', '25', '26', '27', '29'] # note there's no 28 in the data
        elif self.data_set_name in ['machine', 'forestfires', 'wine']:
            return ['regression_output']
        else:
            raise Exception('Unknown/unhandled data set name: %s' % str(self.data_set_name))


    '''
    return corresponding class/regression value for given vector of activations

    INPUT:
        - activations: vector of activations for output layer

    OUTPUT:
        - return single class/regression value determined by activations
    '''
    def get_output_val(self, activations):
        self.logger.log('DEBUG', 'activations: %s' % str(activations))
        if self.CLASSIFICATION:
            ordered_output_nodes = self.get_ordered_output_nodes()
            # return class value for node with highest activation
            return ordered_output_nodes[np.argmax(activations)]
        elif self.REGRESSION:
            # for regression, the output is always a single value in range [0,1]
            # BUG: activations var here should be a single vector, not a list of vectors...
            return activations[0][0]


    '''
    log and return result - compare prediction to actual

    INPUT:
        - idx: index of test instance
        - output: network output value
        - actual: actual (expected) class/regression value from data set

    OUTPUT:
        - return error value based on difference between output/actual
    '''
    def handle_prediction(self, idx, output, actual):
        if self.CLASSIFICATION:
            self.logger.log('INFO', '[test point idx: %s, output: %s, actual: %s]' \
                                        % (str(idx), str(output), str(actual)) \
                                        + (' --> WRONG!' if output != actual else ''))

            # return 0/1 correctness result to list of results
            return int(output == actual)

        elif self.REGRESSION:
            self.logger.log('INFO', '[test point idx: %s, output: %s, actual: %s]' \
                                        % (str(idx), str(round(output, 5)), str(round(actual, 5))) \
                                        + (' --> DIFF: %s' % str(round(abs(output - actual), 5))))

            # return absolute value of error to list of results
            return abs(output - actual)
        else:
            raise Exception('Unable to handle prediction value - unknown experiment type.')


    # ACTIVATION FUNCTIONS


    # return sigmoid of each value in z vector
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))


    # return derivative of sigmoid of each value in z vector
    def d_sigmoid(self, z):
        try:
            # will throw exception if not numpy column vector already
            self.logger.log('VERBOSE', 'z.shape: %s' % str(z.shape))
        except:
            # convert to numpy column vector since we need that form
            z = np.array(z).reshape(len(z), 1)
            self.logger.log('DEBUG', 'converted z in d_sigmoid: %s, shape: %s' % (str(z), str(z.shape)))

        return self.sigmoid(z) * (1.0 - self.sigmoid(z))


    # return rectified linear unit of each value in z vector
    def relu(self, z):
        return np.array([((abs(z_val) + z_val) / 2.0) for z_val in z]).reshape(len(z), 1)


    # return derivative of rectified linear unit of each value in z vector
    def d_relu(self, z):
        # positive values = 1 , negative values (or zero) = 0
        z[z > 0] = 1.0
        z[z <= 0] = 0.0
        return z


    # return hyperbolic tangent of each value in z vector
    def tanh(self, z):
        return np.tanh(z)


    # return derivative of hyperbolic tangent of each value in z vector
    def d_tanh(self, z):
        return 1.0 - np.tanh(z)**2.0


    '''
    generic squishification function - keys off the activation functions list specified at class level
        - also does derivative of squishification function when derivative arg is set to True

    INPUT:
        - z: value to squishify - map to some other value determined by activation function
        - layer_idx: index of layer to get corresponding activation function
        - derivative: boolean indicating whether to return the derivative of the activation function

    OUTPUT:
        - return the activation value for the given z value (or its derivative value)
            - layer index determines which activation function to use
                - possible activation functions : (d_)sigmoid, (d_)relu, (d_)tanh
    '''
    def squish(self, z, layer_idx, derivative):
        layer_activation_func = self.layer_activation_funcs[layer_idx]
        if layer_activation_func == 'sigmoid':
            return self.d_sigmoid(z) if derivative else self.sigmoid(z)
        elif layer_activation_func == 'relu':
            return self.d_relu(z) if derivative else self.relu(z)
        elif layer_activation_func == 'tanh':
            return self.d_tanh(z) if derivative else self.tanh(z)
        else:
            raise Exception('ERROR: must specify activation function for layer: %s.' % str(layer_idx))



# EXECUTE SCRIPT


if __name__ == '__main__':

    print('\nrunning base NeuralNetwork...\n')



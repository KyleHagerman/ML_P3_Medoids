#!/usr/bin/env python3


# IMPORTS


import sys
# add following directories to class path
sys.path.append('../../../data')
sys.path.append('../../networks')
sys.path.append('../../logging')
sys.path.append('../../../../project-2/scripts/data_api')
sys.path.append('../../../../project-2/scripts/preprocessing')
sys.path.append('../../../../project-2/scripts/cross_validator')
sys.path.append('../../../../project-2/scripts/utilities')

from neural_network import NeuralNetwork
from logger import Logger

import unittest
import numpy as np
import pandas as pd


# unit tests for neural_network script


class NeuralNetworkTests(unittest.TestCase):


	# SETUP
	

	@classmethod
	def setUpClass(self):
		self.logger = Logger('ERROR')
		self.neural_network = NeuralNetwork('segmentation', [19, 5, 7])


	@classmethod
	def tearDownClass(self):
		pass


	# TESTS


	def test_sigmoid_with_vector(self):
		z_vector = np.array([1, 2, 3, 4, 5])
		sigmoid_vector = self.neural_network.sigmoid(z_vector)
		self.assertTrue(len(sigmoid_vector) == 5)


	def test_sigmoid_with_vector_reshape(self):
		z_vector = np.array([0.46915773737681854, 1.029212413206881, \
			-2.7460326850612837, -1.4603935532594379, 1.615041766957284]).reshape(5, 1)
		self.logger.log('DEBUG', '\nz_vector: %s, shape: %s' % (str(z_vector), str(z_vector.shape)))

		sigmoid_vector = self.neural_network.sigmoid(z_vector)
		self.logger.log('DEBUG', 'sigmoid_vector: %s, shape: %s' % (str(sigmoid_vector), str(sigmoid_vector.shape)))
		self.assertTrue(len(sigmoid_vector) == 5)

		other_z_vector = np.array([0.9675410739703252, 2.509336631976015, \
			-0.6884017803838307, -1.1750065429515877, -1.6400959178396506]).reshape(5, 1)
		self.logger.log('DEBUG', '\nother_z_vector: %s, shape: %s' % (str(other_z_vector), str(other_z_vector.shape)))

		other_sigmoid_vector = self.neural_network.sigmoid(other_z_vector)
		self.logger.log('DEBUG', 'other_sigmoid_vector: %s, shape: %s' % (str(other_sigmoid_vector), str(other_sigmoid_vector.shape)))
		self.assertTrue(len(other_sigmoid_vector) == 5)


	def test_get_expected_output(self):
		expected_output_vec = self.neural_network.get_expected_output('PATH')
		self.logger.log('DEBUG', 'expected_output_vec: %s' % str(expected_output_vec))
		self.assertTrue(len(expected_output_vec) == 7) # there are 7 possible classes in segmentation data
		self.assertTrue(sum(expected_output_vec) == 1) # only one value should be 1, rest are 0


	def test_get_output_class_val(self):
		# simple test with an obvious activation value diff
		mock_activations = np.array([0, 0, 1, 0, 0, 0, 0]).reshape(7, 1)
		output_class_val = self.neural_network.get_output_val(mock_activations)
		self.assertTrue(output_class_val == 'FOLIAGE')
		# more realistic test with less obvious activation value diff
		mock_activations = np.array([0.25, 0.79, 0.01, 0.5, 0.4, 0.35, 0.81]).reshape(7, 1)
		output_class_val = self.neural_network.get_output_val(mock_activations)
		self.assertTrue(output_class_val == 'WINDOW')
		# test when max activation value is shared between more than one node
		mock_activations = np.array([0.25, 0.79, 0.01, 0.5, 0.4, 0.35, 0.79]).reshape(7, 1)
		output_class_val = self.neural_network.get_output_val(mock_activations)
		# by default, it will choose the first one with highest activation value
		# TODO: change this so it randomly chooses out of all the nodes with max activation
		self.assertTrue(output_class_val == 'CEMENT')


	def test_shuffling_data_frame(self):
		# full test data frame: 7 rows, 2 attribute columns, one numeric, one categorical
		test_data_frame = pd.DataFrame({'a':[1, 2, 2, 4, 5, 5, 5], \
			'b':['cat', 'cat', 'dog', 'wolf', 'monkey', 'dog', 'cat'], \
			'CLASS':['class1', 'class2', 'class2', 'class1', 'class2', 'class1', 'class1']})

		self.logger.log('DEBUG', 'regular test_data_frame: \n%s\n' % str(test_data_frame))
		shuffled_test_data_frame = test_data_frame.sample(frac=1)
		self.logger.log('DEBUG', 'shuffled test_data_frame: \n%s\n' % str(shuffled_test_data_frame))
		shuffled_test_data_frame = test_data_frame.sample(frac=1)
		self.logger.log('DEBUG', 'shuffled x2 test_data_frame: \n%s\n' % str(shuffled_test_data_frame))


	def test_zip_with_enumerate(self):
		vals_1 = ['foo', 'bar', 'baz']
		vals_2 = ['wow', 'whoa', 'who']
		for idx, val in enumerate(zip(vals_1, vals_2)):
			self.logger.log('DEBUG', 'idx: %s, zip_val_1: %s, zip_val_2: %s' % (str(idx), str(val[0]), str(val[1])))


	def test_relu(self):
		z = np.array([-1.3, 2.56, 3.1, -0.78, 0.25]).reshape(5, 1)
		self.logger.log('DEBUG', '\nz: %s' % str(z))

		relu_z = self.neural_network.relu(z)
		self.logger.log('DEBUG', 'relu_z: %s\n' % str(relu_z))

		negative_vals = relu_z[relu_z < 0]
		# assert that all values are positive in relu_z list
		self.assertTrue(len(negative_vals) == 0)


	def test_d_relu(self):
		z_vals = np.array([5, -4, 3, -2, 1]).reshape(5, 1)
		# positive values = 1, negative values = 0
		expected_d_relu = np.array([1, 0, 1, 0, 1]).reshape(5, 1)
		self.assertTrue(np.array_equal(self.neural_network.d_relu(z_vals), expected_d_relu))



# EXECUTE TEST SCRIPT


# run the test script
if __name__ == '__main__':
    unittest.main()



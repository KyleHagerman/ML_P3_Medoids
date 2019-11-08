#!/usr/bin/env python3

import sys
# add data_api directory to class path
sys.path.append('../../data_api')
sys.path.append('../../algorithms')
sys.path.append('../../utilities')
sys.path.append('../../cross_validator')
sys.path.append('../../preprocessing')

import unittest
from data_api import DataApi
from k_nearest_neighbor import KNN


class KNNTests(unittest.TestCase):


	# SETUP
	

	@classmethod
	def setUpClass(self):
		self.DEBUG = False
		self.METRICS = False

		# construct DataApi instance with path prefix to data directory (relative from here)
		self.data_api_impl = DataApi('../../../data/')


		self.knn_impl = KNN()


	@classmethod
	def tearDownClass(self):
		pass
		

	# TESTS

	'''
	# abalone data prediction
	def test_predict_abalone(self):
		abalone_data = self.data_api_impl.get_raw_data_frame('abalone')
		abalone_10_prediction = self.knn_impl.predict(abalone_data, 10, k=5)
		self.verify_prediction('abalone', abalone_10_prediction)


	# car data prediction
	def test_predict_car(self):
		car_data = self.data_api_impl.get_raw_data_frame('car')
		car_10_prediction = self.knn_impl.predict(car_data, 10, k=5)
		self.verify_prediction('car', car_10_prediction)


	# forestfires data prediction
	def test_predict_forestfires(self):
		forestfires_data = self.data_api_impl.get_raw_data_frame('forestfires')
		forestfires_10_prediction = self.knn_impl.predict(forestfires_data, 10, k=5)
		self.verify_prediction('forestfires', forestfires_10_prediction)


	# machine data prediction
	def test_predict_machine(self):
		machine_data = self.data_api_impl.get_raw_data_frame('machine')
		machine_10_prediction = self.knn_impl.predict(machine_data, 10, k=5)
		self.verify_prediction('machine', machine_10_prediction)


	# segmentation data prediction
	def test_predict_segmentation(self):
		segmentation_data = self.data_api_impl.get_raw_data_frame('segmentation')
		segmentation_10_prediction = self.knn_impl.predict(segmentation_data, 10, k=5)
		self.verify_prediction('segmentation', segmentation_10_prediction)


	# wine data prediction
	def test_predict_wine(self):
		wine_data = self.data_api_impl.get_raw_data_frame('wine')
		wine_10_prediction = self.knn_impl.predict(wine_data, 10, k=5)
		self.verify_prediction('wine', wine_10_prediction)


	# test get nearest neighbors
	def test_get_nearest_neighbors(self):
		abalone_data = self.data_api_impl.get_raw_data_frame('abalone')
		self.assertTrue(abalone_data is not None)

		# verify size of returned list matches k value (returns k nearest neighbors)
		self.assertTrue(len(self.knn_impl.get_nearest_neighbors(abalone_data, 0, k=5)) == 5)
		self.assertTrue(len(self.knn_impl.get_nearest_neighbors(abalone_data, 0, k=50)) == 50)
		self.assertTrue(len(self.knn_impl.get_nearest_neighbors(abalone_data, 0, k=100)) == 100)


	# HELPER METHODS


	def verify_prediction(self, data_set_name, prediction):
		if data_set_name == 'abalone':
			self.assertTrue(prediction in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, \
										   12, 13, 14, 15, 16, 17, 18, 19, 20, \
										   21, 22, 23, 24, 25, 26, 27, 28, 29])
		elif data_set_name == 'car':
			self.assertTrue(prediction in ['unacc', 'acc', 'good', 'v-good'])
		elif data_set_name == 'forestfires':
			self.assertTrue(isinstance(prediction, float))
		elif data_set_name == 'machine':
			self.assertTrue(isinstance(prediction, int))
		elif data_set_name == 'segmentation':
			self.assertTrue(prediction in ['brickface', 'sky', 'foliage', 'cement', 'window', 'path', 'grass'])
		elif data_set_name == 'wine':
			self.assertTrue(prediction in [1, 2, 3])
	'''

	'''
	def test_get_nn_mode_no_mode_statistics_error(self):
		nn_labels = [1, 2, 3]
		nn_mode = self.knn_impl.get_nn_mode(nn_labels)
		print(nn_mode)
		self.assertTrue(nn_mode in nn_labels)
	'''


	def test_get_nn_modes_numbers_unique(self):
		nn_labels = [7, 4, 1, 4, 3, 8, 2]
		nn_mode = self.knn_impl.get_nn_mode(nn_labels)
		self.assertTrue(nn_mode is 4) # unique mode is 4


	def test_get_nn_modes_numbers_multiple(self):
		nn_labels = [7, 4, 1, 4, 7, 8, 9, 1]
		nn_mode = self.knn_impl.get_nn_mode(nn_labels)
		self.assertTrue(nn_mode in [1, 4, 7]) # modes = 1, 4, 7


	def test_get_nn_modes_strings_unique(self):
		nn_labels = ['label1', 'label2', 'label1']
		nn_mode = self.knn_impl.get_nn_mode(nn_labels)
		self.assertTrue(nn_mode is 'label1') # unique mode is 'label1'


	def test_get_nn_modes_strings_multiple(self):
		nn_labels = ['label1', 'label2', 'label1', 'label2', 'label3']
		nn_mode = self.knn_impl.get_nn_mode(nn_labels)
		self.assertTrue(nn_mode in ['label1', 'label2']) # modes = 'label1', 'label2'



# run the test script
if __name__ == '__main__':
    unittest.main()


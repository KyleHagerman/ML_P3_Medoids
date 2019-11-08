#!/usr/bin/env python3

import sys
# add following directories to class path
sys.path.append('../../data_api')
sys.path.append('../../utilities')

import unittest
from data_api import DataApi
from utils import DistanceFunctions

class DistanceFunctionsTests(unittest.TestCase):


	# SETUP
	

	@classmethod
	def setUpClass(self):
		self.DEBUG = False
		self.METRICS = False

		# construct DataApi instance with path prefix to data directory (relative from here)
		self.data_api_impl = DataApi('../../../data/')

		self.distance_functions_impl = DistanceFunctions()


	@classmethod
	def tearDownClass(self):
		pass
		

	# TESTS


	# test get manhattan distance
	def test_get_manhattan_distance(self):
		pass


	# test get euclidean distance
	def test_get_euclidean_distance(self):
		abalone_data = self.data_api_impl.get_raw_data_frame('abalone')
		self.assertTrue(abalone_data is not None)

		distance_1_2 = self.distance_functions_impl.get_euclidean_distance(abalone_data[0,:], abalone_data[1,:])
		print('distance_1_2: ' + str(distance_1_2))



# run the test script
if __name__ == '__main__':
    unittest.main()


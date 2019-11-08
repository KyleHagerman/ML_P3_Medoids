#!/usr/bin/env python3

import sys
# add data_api directory to class path
sys.path.append('../../data_api')

import unittest
from data_api import DataApi

class DataApiTests(unittest.TestCase):


	# SETUP
	

	@classmethod
	def setUpClass(self):
		self.DEBUG = False
		self.METRICS = False

		# construct DataApi instance with path prefix to data directory (relative from here)
		self.data_api_impl = DataApi('../../../data/')


	@classmethod
	def tearDownClass(self):
		pass
		

	# TESTS


	# test abalone data retrieval, number of rows/columns
	def test_get_abalone_data(self):
		abalone_data = self.data_api_impl.get_raw_data_frame('abalone')
		self.assertTrue(abalone_data is not None)
		self.assertTrue(abalone_data.shape[0] == 4177) # 4177 rows in abalone data matrix
		self.assertTrue(abalone_data.shape[1] == 9)  # 9 attribute columns in abalone data matrix


	# test car data retrieval, number of rows/columns
	def test_get_car_data(self):
		car_data = self.data_api_impl.get_raw_data_frame('car')
		self.assertTrue(car_data is not None)
		self.assertTrue(car_data.shape[0] == 1728) # 1728 rows in car data matrix
		self.assertTrue(car_data.shape[1] == 7)  # 7 attribute columns in car data matrix


	# test forestfires data retrieval, number of rows/columns
	def test_get_forestfires_data(self):
		forestfires_data = self.data_api_impl.get_raw_data_frame('forestfires')
		self.assertTrue(forestfires_data is not None)
		self.assertTrue(forestfires_data.shape[0] == 518) # 518 rows in forestfires data matrix
		self.assertTrue(forestfires_data.shape[1] == 13)  # 13 attribute columns in forestfires data matrix


	# test machine data retrieval, number of rows/columns
	def test_get_machine_data(self):
		machine_data = self.data_api_impl.get_raw_data_frame('machine')
		self.assertTrue(machine_data is not None)
		self.assertTrue(machine_data.shape[0] == 209) # 209 rows in machine data matrix
		self.assertTrue(machine_data.shape[1] == 10)  # 10 attribute columns in machine data matrix


	# test segmentation data retrieval, number of rows/columns
	def test_get_segmentation_data(self):
		segmentation_data = self.data_api_impl.get_raw_data_frame('segmentation')
		self.assertTrue(segmentation_data is not None)
		self.assertTrue(segmentation_data.shape[0] == 213) # 213 rows in segmentation data matrix
		self.assertTrue(segmentation_data.shape[1] == 20)  # 20 attribute columns in segmentation data matrix


	# test wine data retrieval, number of rows/columns
	def test_get_wine_data(self):
		wine_data = self.data_api_impl.get_raw_data_frame('wine')
		self.assertTrue(wine_data is not None)
		self.assertTrue(wine_data.shape[0] == 6497) # 6497 rows in wine data matrix
		self.assertTrue(wine_data.shape[1] == 12)  # 12 attribute columns in wine data matrix


# run the test script
if __name__ == '__main__':
    unittest.main()


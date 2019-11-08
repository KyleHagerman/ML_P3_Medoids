#!/usr/bin/env python3


# IMPORTS


import sys
# add below directories to class path
sys.path.append('../../data_api')
sys.path.append('../../cross_validator')
sys.path.append('../../preprocessing')
sys.path.append('../../utilities')

import unittest
import pandas as pd
from data_api import DataApi
from cross_validator import CrossValidator
from preprocessor import Preprocessor


# unit tests for cross_validator script


class CrossValidatorTests(unittest.TestCase):


	# SETUP
	

	@classmethod
	def setUpClass(self):
		self.DEBUG = False
		self.METRICS = False

		self.data_api_impl = DataApi('../../../data/')
		self.cross_validator_impl = CrossValidator()
		self.preprocessor_impl = Preprocessor()


	@classmethod
	def tearDownClass(self):
		pass
		

	# TESTS

	'''
	# test get indexes list for abalone data
	def test_get_indexes_list_abalone_data(self):
		abalone_data = self.data_api_impl.get_raw_data_frame('abalone')
		self.assertTrue(abalone_data is not None)
		abalone_indexes = self.cross_validator_impl.get_indexes_list(abalone_data)
		self.assertTrue(len(abalone_indexes) == 4177) # 4177 rows in abalone data frame
		for i in range(1, 10):
			self.assertTrue(abalone_indexes.count(i) == 417) # each subset has 417 rows
		self.assertTrue(abalone_indexes.count(10) == 424) # last subset has 417 + remaining...


	# test get indexes list for car data
	def test_get_indexes_list_car_data(self):
		car_data = self.data_api_impl.get_raw_data_frame('car')
		self.assertTrue(car_data is not None)
		car_indexes = self.cross_validator_impl.get_indexes_list(car_data)
		self.assertTrue(len(car_indexes) == 1728) # 1728 rows in car data frame
		for i in range(1, 10):
			self.assertTrue(car_indexes.count(i) == 172) # each subset has 172 rows
		self.assertTrue(car_indexes.count(10) == 180) # last subset has 172 + remaining...


	# test get indexes list for forest fires data
	def test_get_indexes_list_ff_data(self):
		ff_data = self.data_api_impl.get_raw_data_frame('forestfires')
		self.assertTrue(ff_data is not None)
		ff_indexes = self.cross_validator_impl.get_indexes_list(ff_data)
		self.assertTrue(len(ff_indexes) == 518) # 518 rows in forest fires data frame
		for i in range(1, 10):
			self.assertTrue(ff_indexes.count(i) == 51) # each subset has 51 rows
		self.assertTrue(ff_indexes.count(10) == 59) # last subset has 51 + remaining...


	# test get indexes list for machine data
	def test_get_indexes_list_machine_data(self):
		machine_data = self.data_api_impl.get_raw_data_frame('machine')
		self.assertTrue(machine_data is not None)
		machine_indexes = self.cross_validator_impl.get_indexes_list(machine_data)
		self.assertTrue(len(machine_indexes) == 209) # 209 rows in machine data frame
		for i in range(1, 10):
			self.assertTrue(machine_indexes.count(i) == 20) # each subset has 20 rows
		self.assertTrue(machine_indexes.count(10) == 29) # last subset has 20 + remaining...


	# test get indexes list for segmentation data
	def test_get_indexes_list_segmentation_data(self):
		segmentation_data = self.data_api_impl.get_raw_data_frame('segmentation')
		self.assertTrue(segmentation_data is not None)
		segmentation_indexes = self.cross_validator_impl.get_indexes_list(segmentation_data)
		self.assertTrue(len(segmentation_indexes) == 213) # 213 rows in segmentation data frame
		for i in range(1, 10):
			self.assertTrue(segmentation_indexes.count(i) == 21) # each subset has 21 rows
		self.assertTrue(segmentation_indexes.count(10) == 24) # last subset has 21 + remaining...


	# test get indexes list for wine data
	def test_get_indexes_list_wine_data(self):
		wine_data = self.data_api_impl.get_raw_data_frame('wine')
		self.assertTrue(wine_data is not None)
		wine_indexes = self.cross_validator_impl.get_indexes_list(wine_data)
		self.assertTrue(len(wine_indexes) == 6497) # 6497 rows in wine data frame
		for i in range(1, 10):
			self.assertTrue(wine_indexes.count(i) == 649) # each subset has 649 rows
		self.assertTrue(wine_indexes.count(10) == 656) # last subset has 649 + remaining...


	# TRAINING SET


	# test get training set 2 with wine data
	def test_get_training_set(self):
		wine_data = self.data_api_impl.get_raw_data_frame('wine')
		wine_data_training_set = self.cross_validator_impl.get_training_set(wine_data, 2)
		self.assertTrue(wine_data_training_set.shape[0] == 5848) # 6497 - 649 rows in test set 2 means 5484 rows in training set
		self.assertTrue(wine_data_training_set.shape[1] == 12) # number of columns does not change


	# TEST SET


	# test get test set (-2) with wine data
	def test_get_test_set(self):
		wine_data = self.data_api_impl.get_raw_data_frame('wine')
		wine_data_test_set = self.cross_validator_impl.get_test_set(wine_data, 2)
		self.assertTrue(wine_data_test_set.shape[0] == 649) # 649 rows in test set 2
		self.assertTrue(wine_data_test_set.shape[1] == 12) # number of columns does not change
	'''

	def test_cv_partitions(self):
		abalone_data = self.data_api_impl.get_raw_data_frame('abalone')
		prep_abalone_data = self.preprocessor_impl.preprocess_raw_data_frame(abalone_data, 'abalone')
		cv_partitions = self.cross_validator_impl.get_cv_partitions(prep_abalone_data)

		self.assertTrue(cv_partitions is not None)

		for partition in cv_partitions:
			train_data_indexes = list(cv_partitions[partition][0].index.values)
			test_data_indexes = list(cv_partitions[partition][1].index.values)
			for test_index in test_data_indexes:
				self.assertTrue(test_index not in train_data_indexes)


# run the test script
if __name__ == '__main__':
    unittest.main()


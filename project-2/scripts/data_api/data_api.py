#!/usr/bin/env python3


# IMPORTS


import sys
# add data directory to class path
sys.path.append('../../data')

import pandas as pd


# CLASS


'''
	This class is responsible for all things related to actually getting the data from the csv .data files.
	It has various utility methods for getting specific rows/columns/values from the data sets.
	We use the "pandas" library as our data structure for holding the data matrices themselves.
'''


class DataApi:


	def __init__(self, relative_path_prefix):
		self.DEBUG = False
		# the relative path prefix is used to properly instantiate the class from anywhere in the application
		# the get_data_frame method below uses it to correctly locate the data set file by relative file path
		self.relative_path_prefix = relative_path_prefix


	# DATA RETRIEVAL METHODS


	'''
	MAIN PUBLIC METHOD - get the full, unchanged (raw) dataframe for the given data set name

	INPUT:
		- data_set_name: name of data set to fetch
		- nrows: number of rows to fetch from data set, default is all rows

	OUTPUT:
		- full dataframe for data set, read from csv .data file in /data/ directory
	'''
	def get_raw_data_frame(self, data_set_name, frac_rows=None):
		# create variable for new (empty) pandas dataframe
		full_data_frame = pd.DataFrame()

		# call handler methods for various data set names
		if data_set_name == 'abalone':
			full_data_frame = self.get_abalone_data()
		elif data_set_name == 'car':
			full_data_frame = self.get_car_data()
		elif data_set_name == 'forestfires':
			full_data_frame = self.get_forestfires_data()
		elif data_set_name == 'machine':
			full_data_frame = self.get_machine_data()
		elif data_set_name == 'segmentation':
			full_data_frame = self.get_segmentation_data()
		elif data_set_name == 'wine':
			full_data_frame = self.get_wine_data()
		else:
			# throw exception if we get a data set name other than the ones above
			raise Exception('ERROR: unknown data_set_name --> ' + str(data_set_name))

		# return full pandas data frame, ready for consumption by preprocessor class
		return full_data_frame if frac_rows is None else full_data_frame.sample(frac=frac_rows)


	# HANDLER METHODS - for each data set


	# get abalone data as pandas data frame - specify column labels
	def get_abalone_data(self):
		abalone_labels = ['Sex', 'Length', 'Diameter', 'Height', 'Whole', 'Shucked', 'Viscera', 'Shell', 'CLASS']
		return self.get_data_frame('abalone.data', abalone_labels)


	# get car data as pandas data frame - specify column labels
	def get_car_data(self):
		car_labels = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'CLASS']
		return self.get_data_frame('car.data', car_labels)


	# get forestfires data as pandas data frame - specify column labels
	def get_forestfires_data(self):
		forestfires_labels = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'CLASS']
		return self.get_data_frame('forestfires.data', forestfires_labels)


	# get machine data as pandas data frame - specify column labels
	def get_machine_data(self):
		machine_labels = ['vendor', 'model', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'CLASS', 'ERP']
		return self.get_data_frame('machine.data', machine_labels)


	# TODO: change this to skip first three rows since they're bogus data
	# get segmentation data as pandas data frame - specify column labels
	def get_segmentation_data(self):
		# easier to reference numbers to get column names from .names file instead of listing them here
		segmentation_labels = ['CLASS', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
										'11', '12', '13', '14', '15', '16', '17', '18', '19']
		return self.get_data_frame('segmentation.data', segmentation_labels)


	# get wine data as pandas data frame - specify column labels
	def get_wine_data(self):
		wine_labels = ['fa', 'va', 'ca', 'rs', 'c', 'fsd', 'tsd', 'd', 'pH', 's', 'a', 'CLASS']
		red_wine_data = self.get_data_frame('winequality-red.csv', wine_labels, separator=';')
		white_wine_data = self.get_data_frame('winequality-white.csv', wine_labels, separator=';')
		wine_data = red_wine_data.append(white_wine_data, ignore_index=True)
		return wine_data


	# HELPER METHODS

	'''
	get list of column labels for given data set, and possibly remove CLASS column
	this method is used by various classes to build subset dataframes and copies of dataframes for whatever

	INPUT:
		- data_set_name: name of data set to get column labels for
		- include_class: boolean indicating whether to include or remove CLASS column

	OUTPUT:
		- return a list of column labels, and remove CLASS column if specified to do so
	'''
	def get_column_labels(self, data_set_name, include_class):
		column_labels = []
		if data_set_name == 'abalone':
			column_labels = ['Sex', 'Length', 'Diameter', 'Height', 'Whole', 'Shucked', 'Viscera', 'Shell', 'CLASS']
		elif data_set_name == 'car':
			column_labels = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'CLASS']
		elif data_set_name == 'forestfires':
			column_labels = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'CLASS']
		elif data_set_name == 'machine':
			column_labels = ['vendor', 'model', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'CLASS', 'ERP']
		elif data_set_name == 'segmentation':
			column_labels = ['CLASS', '1', '2', '3', '4', '5', '6', '7', '8', '9', \
				'10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
		elif data_set_name == 'wine':
			column_labels = ['fa', 'va', 'ca', 'rs', 'c', 'fsd', 'tsd', 'd', 'pH', 's', 'a', 'CLASS']
		else:
			# throw exception if given data set name other than the ones above
			raise Exception('ERROR: unknown data_set_name --> ' + str(data_set_name))

		# return a list of column labels, and remove CLASS column if specified to do so
		return column_labels if include_class else list(filter(lambda lbl: lbl != 'CLASS', column_labels))


	# get pandas data frame consisting of all rows from data_set_name with given class value
	def get_data_frame_for_class(self, data_set_name, class_value):
		full_data_frame = pd.DataFrame()

		# set match var to all indexes that have the given class value
		match = full_data_frame['CLASS'] == class_value
		return full_data_frame[match]


	# return data frame of all rows with given class value
	def get_all_class_instances(self, data_frame, class_value):
		match = data_frame['CLASS'] == class_value
		return data_frame[match]


	# generic method for getting class data frame from full data frame
	def get_class_data_frame(self, full_data_frame, class_value):
		match = full_data_frame['CLASS'] == class_value
		return full_data_frame[match]


	# method for getting full data frame given data set name
	def get_full_data_frame(self, data_set_name):
		full_data_frame = pd.DataFrame()

		return full_data_frame


	# simple method for getting the class value of a given row in a data set
	def get_class_value_for_data_frame_row(self, data_frame, row_index):
		return data_frame.loc[row_index, 'CLASS']


	# method for getting the class value for a given row given just the data set name
	def get_class_value_for_row(self, data_frame_name, row_index):
		full_data_frame = self.get_full_data_frame(data_frame_name)
		return full_data_frame.loc[row_index, 'CLASS']


	# FILE READING METHODS


	# read csv file at file name arg and return data matrix as pandas data frame
	def get_data_frame(self, data_set, column_names, *args, **kwargs):
		# get file path for data set using relative path prefix and data set name
		data_set_file_name = self.relative_path_prefix + data_set
		# read in separator as optional third arg
		separator = kwargs.get('separator', None)
		# default to using a comma as separator if not overridden
		delimiter = separator if separator is not None else ','
		# call pandas library method to return pandas dataframe read from csv file
		return pd.read_csv(data_set_file_name, names=column_names, sep=delimiter)



# EXECUTE SCRIPT


if __name__ == "__main__":

	print('running data_api...')

	datalayer = DataApi('../../data/')

	#print(datalayer.get_column_labels('wine', True))
	#print(datalayer.get_column_labels('wine', False))

	abalone_data = datalayer.get_raw_data_frame('abalone')
	print('DATA:')
	print(abalone_data)

	print('\n\nrow')
	print(abalone_data.iloc[2, :])

	'''
	#wine_data = datalayer.get_wine_data()
	#print(wine_data)
	#print(wine_data.shape)
	#print(wine_data.loc[4898,:])
	'''

#!/usr/bin/env python3

# !!! NOTE: this script has to been ran from the data_api root directory on command line !!!


# IMPORTS


import sys
# add data_api directory to class path
sys.path.append('../data_api')
sys.path.append('../../../data')

from data_api import DataApi


# CLASS


'''
	Simple utility class for viewing data sets from command line.
'''


class ViewData():


	def __init__(self):
		self.DEBUG = False
		# construct DataApi instance with path prefix to data directory (relative from here)
		self.data_api_impl = DataApi('../../data/')


	def view_abalone_data(self):
		print('\nABALONE DATA:\n')
		print(self.data_api_impl.get_raw_data_frame('abalone'))


	def view_car_data(self):
		print('\nCAR DATA:\n')
		print(self.data_api_impl.get_raw_data_frame('car'))


	def view_forestfires_data(self):
		print('\nFORESTFIRES DATA:\n')
		print(self.data_api_impl.get_raw_data_frame('forestfires'))


	def view_machine_data(self):
		print('\nMACHINE DATA:\n')
		print(self.data_api_impl.get_raw_data_frame('machine'))


	def view_segmentation_data(self):
		print('\nSEGMENTATION DATA:\n')
		print(self.data_api_impl.get_raw_data_frame('segmentation'))


	def view_wine_data(self):
		print('\nWINE DATA:\n')
		print(self.data_api_impl.get_raw_data_frame('wine'))


# EXECUTE SCRIPT


# run the view_data script to view data matrices from command line for easy reference
if __name__ == '__main__':

	view_data_impl = ViewData()

	view_data_impl.view_abalone_data()
	view_data_impl.view_car_data()
	view_data_impl.view_forestfires_data()
	view_data_impl.view_machine_data()
	view_data_impl.view_segmentation_data()
	view_data_impl.view_wine_data()
	

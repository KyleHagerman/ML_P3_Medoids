import unittest


# EXECUTE SCRIPT


if __name__ == "__main__":

	print('running all tests...')

	loader = unittest.TestLoader()
	start_dir = 'tests'
	suite = loader.discover(start_dir)

	runner = unittest.TextTestRunner()
	runner.run(suite)

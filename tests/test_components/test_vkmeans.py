import unittest

from tests.utils import \
    StandardPreprocessingTest, \
    generate_preprocessor_attributes
from hpsklearn import colkmeans, \
    gaussian_nb


class TestColkmeansPreprocessing(StandardPreprocessingTest):
    """
    Class for _data preprocessing testing
    """


generate_preprocessor_attributes(
    TestClass=TestColkmeansPreprocessing,
    preprocessor_list=[colkmeans],
    classifier=gaussian_nb,
)


if __name__ == '__main__':
    unittest.main()

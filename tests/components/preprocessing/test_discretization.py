import unittest

from tests.utils import \
    StandardPreprocessingTest, \
    generate_preprocessor_attributes
from hpsklearn import \
    k_bins_discretizer, \
    gaussian_nb


class TestDiscretizationPreprocessing(StandardPreprocessingTest):
    """
    Class for _discretization preprocessing testing
    """


generate_preprocessor_attributes(
    TestClass=TestDiscretizationPreprocessing,
    preprocessor_list=[k_bins_discretizer],
    classifier=gaussian_nb,
)


if __name__ == '__main__':
    unittest.main()

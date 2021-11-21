import unittest

from hpsklearn.tests.utils import \
    StandardPreprocessingTest, \
    generate_preprocessor_test_attributes
from hpsklearn import \
    polynomial_features, \
    spline_transformer, \
    gaussian_nb


class TestPolynomialPreprocessing(StandardPreprocessingTest):
    """
    Class for _polynomial preprocessing testing
    """


# List of preprocessors to test
preprocessors = [
    polynomial_features,
    spline_transformer
]


generate_preprocessor_test_attributes(
    TestClass=TestPolynomialPreprocessing,
    preprocessor_list=preprocessors,
    classifier=gaussian_nb,
)


if __name__ == '__main__':
    unittest.main()

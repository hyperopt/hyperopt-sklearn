import unittest

from hpsklearn import \
    pca, \
    gaussian_nb
from tests.utils import \
    StandardPreprocessingTest, \
    generate_preprocessor_attributes


class TestPCA(StandardPreprocessingTest):
    """
    Class for _pca preprocessing testing
    """


generate_preprocessor_attributes(
    TestClass=TestPCA,
    preprocessor_list=[pca],
    classifier=gaussian_nb,
)


if __name__ == '__main__':
    unittest.main()

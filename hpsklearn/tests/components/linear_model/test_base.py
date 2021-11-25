import unittest

from hyperopt import rand

from hpsklearn.tests.utils import \
    StandardRegressorTest, \
    generate_attributes
from hpsklearn import hyperopt_estimator, \
    linear_regression


class TestBaseRegression(StandardRegressorTest):
    """
    Class for _base regression testing
    """


generate_attributes(
    TestClass=TestBaseRegression,
    fn_list=[linear_regression],
    is_classif=False
)


if __name__ == '__main__':
    unittest.main()

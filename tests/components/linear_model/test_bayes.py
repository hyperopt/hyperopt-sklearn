import unittest

from tests.utils import \
    StandardRegressorTest, \
    generate_attributes
from hpsklearn import \
    bayesian_ridge, \
    ard_regression


class TestBayesRegression(StandardRegressorTest):
    """
    Class for _bayes regression testing
    """


generate_attributes(
    TestClass=TestBayesRegression,
    fn_list=[bayesian_ridge, ard_regression],
    is_classif=False
)


if __name__ == '__main__':
    unittest.main()

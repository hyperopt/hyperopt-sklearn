import unittest

from hpsklearn import \
    lightgbm_classification, \
    lightgbm_regression
from hpsklearn.tests.utils import \
    StandardRegressorTest, \
    StandardClassifierTest, \
    generate_test_attributes


class TestLightGBMClassifier(StandardClassifierTest):
    """
    Class for _lightgbm classification testing
    """


class TestLightGBMRegression(StandardRegressorTest):
    """
    Class for _lightgbm regression testing
    """


generate_test_attributes(
    TestClass=TestLightGBMClassifier,
    fn_list=[lightgbm_classification],
    is_classif=True,
)


generate_test_attributes(
    TestClass=TestLightGBMRegression,
    fn_list=[lightgbm_regression],
    is_classif=False,
)


if __name__ == '__main__':
    unittest.main()

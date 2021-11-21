import unittest

from hpsklearn import \
    xgboost_classification, \
    xgboost_regression
from hpsklearn.tests.utils import \
    StandardRegressorTest, \
    StandardClassifierTest, \
    generate_test_attributes


class TestXGBoostClassifier(StandardClassifierTest):
    """
    Class for _xgboost classification testing
    """


class TestXGBoostRegression(StandardRegressorTest):
    """
    Class for _xgboost regression testing
    """


generate_test_attributes(
    TestClass=TestXGBoostClassifier,
    fn_list=[xgboost_classification],
    is_classif=True,
)


generate_test_attributes(
    TestClass=TestXGBoostRegression,
    fn_list=[xgboost_regression],
    is_classif=False,
)


if __name__ == '__main__':
    unittest.main()

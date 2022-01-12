import unittest

from hpsklearn import \
    xgboost_classification, \
    xgboost_regression
from tests.utils import \
    StandardRegressorTest, \
    StandardClassifierTest, \
    generate_attributes


class TestXGBoostClassifier(StandardClassifierTest):
    """
    Class for _xgboost classification testing
    """


class TestXGBoostRegression(StandardRegressorTest):
    """
    Class for _xgboost regression testing
    """


generate_attributes(
    TestClass=TestXGBoostClassifier,
    fn_list=[xgboost_classification],
    is_classif=True,
)


generate_attributes(
    TestClass=TestXGBoostRegression,
    fn_list=[xgboost_regression],
    is_classif=False,
)


if __name__ == '__main__':
    unittest.main()

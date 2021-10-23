import unittest

from hpsklearn.tests.utils import \
    StandardClassifierTest, \
    StandardRegressorTest, \
    generate_test_attributes
from hpsklearn import \
    bagging_classifier, \
    bagging_regressor


class TestBaggingClassification(StandardClassifierTest):
    """
    Class for _bagging classification testing
    """


class TestBaggingRegression(StandardRegressorTest):
    """
    Class for _bagging regression testing
    """


generate_test_attributes(
    TestClass=TestBaggingClassification,
    fn_list=[bagging_classifier],
    is_classif=True
)


generate_test_attributes(
    TestClass=TestBaggingRegression,
    fn_list=[bagging_regressor],
    is_classif=False
)


if __name__ == '__main__':
    unittest.main()

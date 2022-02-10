import unittest

from hpsklearn import \
    ridge, \
    ridge_cv, \
    ridge_classifier, \
    ridge_classifier_cv
from tests.utils import \
    StandardClassifierTest, \
    StandardRegressorTest, \
    generate_attributes


class TestRidgeClassification(StandardClassifierTest):
    """
    Class for _ridge classification testing
    """


class TestRidgeRegression(StandardRegressorTest):
    """
    Class for _ridge regression testing
    """


generate_attributes(
    TestClass=TestRidgeClassification,
    fn_list=[ridge_classifier, ridge_classifier_cv],
    is_classif=True
)


generate_attributes(
    TestClass=TestRidgeRegression,
    fn_list=[ridge, ridge_cv],
    is_classif=False
)


if __name__ == '__main__':
    unittest.main()

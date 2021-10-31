import unittest

from hpsklearn import \
    passive_aggressive_classifier, \
    passive_aggressive_regressor
from hpsklearn.tests.utils import \
    StandardRegressorTest, \
    StandardClassifierTest, \
    generate_test_attributes


class TestPassiveAggressiveClassifier(StandardClassifierTest):
    """
    Class for _passive_aggressive classification testing
    """


class TestPassiveAggressiveRegression(StandardRegressorTest):
    """
    Class for _passive_aggressive regression testing
    """


generate_test_attributes(
    TestClass=TestPassiveAggressiveClassifier,
    fn_list=[passive_aggressive_classifier],
    is_classif=True,
)


generate_test_attributes(
    TestClass=TestPassiveAggressiveRegression,
    fn_list=[passive_aggressive_regressor],
    is_classif=False,
)


if __name__ == '__main__':
    unittest.main()

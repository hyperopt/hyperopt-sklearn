import unittest

from hpsklearn import \
    dummy_classifier, \
    dummy_regressor

from hpsklearn.tests.utils import \
    StandardRegressorTest, \
    StandardClassifierTest, \
    generate_test_attributes


class TestDummyClassifier(StandardClassifierTest):
    """
    Class for _dummy classification testing
    """


class TestDummyRegression(StandardRegressorTest):
    """
    Class for _dummy regression testing
    """


generate_test_attributes(
    TestClass=TestDummyClassifier,
    fn_list=[dummy_classifier],
    is_classif=True,
)


generate_test_attributes(
    TestClass=TestDummyRegression,
    fn_list=[dummy_regressor],
    is_classif=False,
)


if __name__ == '__main__':
    unittest.main()

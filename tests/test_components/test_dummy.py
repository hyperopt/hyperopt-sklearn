import unittest

from hpsklearn import \
    dummy_classifier, \
    dummy_regressor

from tests.utils import \
    StandardRegressorTest, \
    StandardClassifierTest, \
    generate_attributes


class TestDummyClassifier(StandardClassifierTest):
    """
    Class for _dummy classification testing
    """


class TestDummyRegression(StandardRegressorTest):
    """
    Class for _dummy regression testing
    """


generate_attributes(
    TestClass=TestDummyClassifier,
    fn_list=[dummy_classifier],
    is_classif=True,
)


generate_attributes(
    TestClass=TestDummyRegression,
    fn_list=[dummy_regressor],
    is_classif=False,
)


if __name__ == '__main__':
    unittest.main()

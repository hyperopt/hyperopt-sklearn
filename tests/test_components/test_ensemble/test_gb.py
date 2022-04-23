import unittest

from hpsklearn import \
    gradient_boosting_classifier, \
    gradient_boosting_regressor
from tests.utils import \
    StandardClassifierTest, \
    StandardRegressorTest, \
    generate_attributes


class TestGradientBoostingClassification(StandardClassifierTest):
    """
    Class for _gb classification testing
    """


class TestGradientBoostingRegression(StandardRegressorTest):
    """
    Class for _gb regression testing
    """


generate_attributes(
    TestClass=TestGradientBoostingClassification,
    fn_list=[gradient_boosting_classifier],
    is_classif=True,
)


generate_attributes(
    TestClass=TestGradientBoostingRegression,
    fn_list=[gradient_boosting_regressor],
    is_classif=False,
)


if __name__ == '__main__':
    unittest.main()

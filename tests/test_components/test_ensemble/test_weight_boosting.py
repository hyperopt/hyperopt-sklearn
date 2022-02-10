import unittest

from hpsklearn import \
    ada_boost_classifier, \
    ada_boost_regressor
from tests.utils import \
    StandardClassifierTest, \
    StandardRegressorTest, \
    generate_attributes


class TestWeightBoostingClassification(StandardClassifierTest):
    """
    Class for _weight_boosting classification testing
    """


class TestWeightBoostingRegression(StandardRegressorTest):
    """
    Class for _weight_boosting regression testing
    """


generate_attributes(
    TestClass=TestWeightBoostingClassification,
    fn_list=[ada_boost_classifier],
    is_classif=True
)


generate_attributes(
    TestClass=TestWeightBoostingRegression,
    fn_list=[ada_boost_regressor],
    is_classif=False
)


if __name__ == '__main__':
    unittest.main()

import unittest

from hpsklearn import \
    mlp_classifier, \
    mlp_regressor
from tests.utils import \
    StandardRegressorTest, \
    StandardClassifierTest, \
    generate_attributes


class TestMLPClassifier(StandardClassifierTest):
    """
    Class for _multilayer_perceptron classification testing
    """


class TestMLPRegression(StandardRegressorTest):
    """
    Class for _multilayer_perceptron regression testing
    """


generate_attributes(
    TestClass=TestMLPClassifier,
    fn_list=[mlp_classifier],
    is_classif=True,
    non_negative_input=True,
    non_negative_output=True
)


generate_attributes(
    TestClass=TestMLPRegression,
    fn_list=[mlp_regressor],
    is_classif=False,
    non_negative_input=True,
    non_negative_output=True
)


if __name__ == '__main__':
    unittest.main()

import unittest

from hpsklearn import \
    mlp_classifier, \
    mlp_regressor
from hpsklearn.tests.utils import \
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
)


generate_attributes(
    TestClass=TestMLPRegression,
    fn_list=[mlp_regressor],
    is_classif=False,
)


if __name__ == '__main__':
    unittest.main()

import unittest

from hpsklearn import gaussian_mixture
from tests.utils import \
    StandardRegressorTest, \
    StandardClassifierTest, \
    generate_attributes


class TestGaussianMixtureClassifier(StandardClassifierTest):
    """
    Class for _gaussian_mixture classification testing
    """


class TestGaussianMixtureRegression(StandardRegressorTest):
    """
    Class for _gaussian_mixture regression testing
    """


generate_attributes(
    TestClass=TestGaussianMixtureClassifier,
    fn_list=[gaussian_mixture],
    is_classif=True,
)


generate_attributes(
    TestClass=TestGaussianMixtureRegression,
    fn_list=[gaussian_mixture],
    is_classif=False,
)


if __name__ == '__main__':
    unittest.main()

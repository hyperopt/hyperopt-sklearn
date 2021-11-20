import unittest

from hpsklearn import gaussian_mixture
from hpsklearn.tests.utils import \
    StandardRegressorTest, \
    StandardClassifierTest, \
    generate_test_attributes


class TestGaussianMixtureClassifier(StandardClassifierTest):
    """
    Class for _gaussian_mixture classification testing
    """


class TestGaussianMixtureRegression(StandardRegressorTest):
    """
    Class for _gaussian_mixture regression testing
    """


generate_test_attributes(
    TestClass=TestGaussianMixtureClassifier,
    fn_list=[gaussian_mixture],
    is_classif=True,
)


generate_test_attributes(
    TestClass=TestGaussianMixtureRegression,
    fn_list=[gaussian_mixture],
    is_classif=False,
    )


if __name__ == '__main__':
    unittest.main()

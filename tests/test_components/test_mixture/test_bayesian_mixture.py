import unittest

from hpsklearn import bayesian_gaussian_mixture
from tests.utils import \
    StandardRegressorTest, \
    StandardClassifierTest, \
    generate_attributes


class TestBayesianGaussianMixtureClassifier(StandardClassifierTest):
    """
    Class for _bayesian_mixture classification testing
    """


class TestBayesianGaussianMixtureRegression(StandardRegressorTest):
    """
    Class for _bayesian_mixture regression testing
    """


generate_attributes(
    TestClass=TestBayesianGaussianMixtureClassifier,
    fn_list=[bayesian_gaussian_mixture],
    is_classif=True,
)


generate_attributes(
    TestClass=TestBayesianGaussianMixtureRegression,
    fn_list=[bayesian_gaussian_mixture],
    is_classif=False,
)


if __name__ == '__main__':
    unittest.main()

import unittest

from hpsklearn import bayesian_gaussian_mixture
from hpsklearn.tests.utils import \
    StandardRegressorTest, \
    StandardClassifierTest, \
    generate_test_attributes


class TestBayesianGaussianMixtureClassifier(StandardClassifierTest):
    """
    Class for _bayesian_mixture classification testing
    """


class TestBayesianGaussianMixtureRegression(StandardRegressorTest):
    """
    Class for _bayesian_mixture regression testing
    """


generate_test_attributes(
    TestClass=TestBayesianGaussianMixtureClassifier,
    fn_list=[bayesian_gaussian_mixture],
    is_classif=True,
)


generate_test_attributes(
    TestClass=TestBayesianGaussianMixtureRegression,
    fn_list=[bayesian_gaussian_mixture],
    is_classif=False,
    )


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np

from hyperopt import rand

from hpsklearn.tests.utils import \
    StandardClassifierTest
from hpsklearn import hyperopt_estimator, \
    bernoulli_nb, \
    categorical_nb, \
    complement_nb, \
    gaussian_nb, \
    multinomial_nb


class TestNaiveBayes(StandardClassifierTest):
    """
    Class for naive_bayes classification testing
    """


# List of naive bayes to test
naive_bayes = [
    bernoulli_nb,
    gaussian_nb
]

# List of non-negative input naive bayes to test
non_negative_naive_bayes = [
    categorical_nb,
    complement_nb,
    multinomial_nb
]


def create_naive_bayes_function(nb_fn):
    """
    Instantiate standard hyperopt estimator model
     'nb_fn' regards the naive bayes that is tested
     fit and score model
    """
    def test_naive_bayes(self):
        model = hyperopt_estimator(
            classifier=nb_fn("classifier"),
            preprocessing=[],
            algo=rand.suggest,
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(self.X_train, self.Y_train)
        model.score(self.X_test, self.Y_test)

    test_naive_bayes.__name__ = f"test_{nb_fn.__name__}"
    return test_naive_bayes


def create_non_negative_naive_bayes_function(nn_nb_fn):
    """
    Instantiate standard hyperopt estimator model
     'nn_nb_fn' regards the naive bayes that is tested
     fit and score model
     ensure non-negative input
    """
    def test_non_negative_naive_bayes(self):
        model = hyperopt_estimator(
            regressor=nn_nb_fn(name="classifier"),
            preprocessing=[],
            algo=rand.suggest,
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(np.abs(self.X_train), self.Y_train)
        model.score(np.abs(self.X_test), self.Y_test)

    test_non_negative_naive_bayes.__name__ = f"test_{nn_nb_fn.__name__}"
    return test_non_negative_naive_bayes


# Create unique naive_bayes testing methods
#  with test_ prefix so that nose can see them
for nb in naive_bayes:
    setattr(
        TestNaiveBayes,
        f"test_{nb.__name__}",
        create_naive_bayes_function(nb)
    )


# Create unique non-negative naive_bayes testing methods
#  with test_ prefix so that nose can see them
for nn_nb in non_negative_naive_bayes:
    setattr(
        TestNaiveBayes,
        f"test_{nn_nb.__name__}",
        create_non_negative_naive_bayes_function(nn_nb)
    )


if __name__ == '__main__':
    unittest.main()

from .._forest import random_forest_classifier, \
    random_forest_regressor, \
    extra_trees_classifier, \
    extra_trees_regressor

import unittest
import numpy as np

from hyperopt import rand
from hpsklearn.estimator import hyperopt_estimator


class TestRegression(unittest.TestCase):
    """
    Class for standard regression testing
     setup of randomly generated data
    """
    def setUp(self):
        np.random.seed(123)
        self.X_train = np.random.randn(1000, 2)
        self.Y_train = self.X_train[:, 0] * 2
        self.X_test = np.random.randn(1000, 2)
        self.Y_test = self.X_test[:, 0] * 2


class TestPoissonRandomForestRegression(unittest.TestCase):
    """
    Class for testing 'poisson' criterion in random forest regression
     setup of randomly generated, non-negative data
    """
    def setUp(self):
        np.random.seed(123)
        self.X_train = np.random.uniform(0, 10, (10, 2))
        self.Y_train = self.X_train[:, 0] * 2
        self.X_test = np.random.uniform(0, 10, (10, 2))
        self.Y_test = self.X_test[:, 0] * 2

    def test_poisson_function(self):
        """
        Instantiate random forest regressor hyperopt estimator model
         define 'criterion' = 'poisson'
         fit and score model
        """
        model = hyperopt_estimator(
            regressor=random_forest_regressor(name="poisson_regressor",
                                              criterion="poisson"),
            preprocessing=[],
            algo=rand.suggest,
            trial_timeout=5.0,
            max_evals=5,
        )
        model.fit(self.X_train, self.Y_train)
        model.score(self.X_test, self.Y_test)

    test_poisson_function.__name__ = f"test_{random_forest_regressor.__name__}"


# List of regressors to test
regressors = [
    random_forest_regressor,
    extra_trees_regressor
]

# List of classifiers to test
classifiers = [
    random_forest_classifier,
    extra_trees_classifier,
]


def create_function(reg_fn):
    def test_regressor(self):
        """
        Instantiate standard hyperopt estimator model
         'req_fn' regards regressor that is tested
         fit and score model
        """
        model = hyperopt_estimator(
            regressor=reg_fn("regressor"),
            preprocessing=[],
            algo=rand.suggest,
            trial_timeout=5.0,
            max_evals=5,
        )
        model.fit(self.X_train, self.Y_train)
        model.score(self.X_test, self.Y_test)

    test_regressor.__name__ = f"test_{reg_fn.__name__}"
    return test_regressor


# Create unique regression testing methods with test_ prefix so that nose can see them
for reg in regressors:
    setattr(
        TestRegression,
        f"test_{reg.__name__}",
        create_function(reg)
    )


if __name__ == '__main__':
    unittest.main()

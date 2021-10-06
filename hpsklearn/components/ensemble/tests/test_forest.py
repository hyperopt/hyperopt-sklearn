from .._forest import \
    random_forest_classifier, \
    random_forest_regressor, \
    extra_trees_classifier, \
    extra_trees_regressor

import unittest
import numpy as np

from hyperopt import rand
from hpsklearn.estimator import hyperopt_estimator


class TestForestRegression(unittest.TestCase):
    """
    Class for _forest regression testing
    """
    def setUp(self):
        """
        Setup of randomly generated data
        """
        np.random.seed(123)
        self.X_train = np.random.randn(1000, 2)
        self.Y_train = self.X_train[:, 0] * 2
        self.X_test = np.random.randn(1000, 2)
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
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(np.abs(self.X_train), np.abs(self.Y_train))
        model.score(np.abs(self.X_test), np.abs(self.Y_test))

    test_poisson_function.__name__ = f"test_{random_forest_regressor.__name__}"


class TestForestClassification(unittest.TestCase):
    """
    Class for _forest classification testing
    """
    def setUp(self):
        """
        Setup of randomly generated data
        """
        np.random.seed(123)
        self.X_train = np.random.randn(1000, 2)
        self.Y_train = (self.X_train[:, 0] > 0).astype('int')
        self.X_test = np.random.randn(1000, 2)
        self.Y_test = (self.X_test[:, 0] > 0).astype('int')


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


def create_regressor_function(reg_fn):
    """
    Instantiate standard hyperopt estimator model
     'reg_fn' regards the regressor that is tested
     fit and score model
    """
    def test_regressor(self):
        model = hyperopt_estimator(
            regressor=reg_fn("regressor"),
            preprocessing=[],
            algo=rand.suggest,
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(self.X_train, self.Y_train)
        model.score(self.X_test, self.Y_test)

    test_regressor.__name__ = f"test_{reg_fn.__name__}"
    return test_regressor


def create_classifier_function(clf_fn):
    """
    Instantiate standard hyperopt estimator model
     'clf_fn' regards the classifier that is tested
     fit and score model
    """
    def test_classifier(self):
        model = hyperopt_estimator(
            regressor=clf_fn("classifier"),
            preprocessing=[],
            algo=rand.suggest,
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(self.X_train, self.Y_train)
        model.score(self.X_test, self.Y_test)

    test_classifier.__name__ = f"test_{clf_fn.__name__}"
    return test_classifier


# Create unique _forest regression testing methods
#  with test_ prefix so that nose can see them
for reg in regressors:
    setattr(
        TestForestRegression,
        f"test_{reg.__name__}",
        create_regressor_function(reg)
    )

# Create unique _forest classification testing methods
#  with test_ prefix so that nose can see them
for clf in classifiers:
    setattr(
        TestForestClassification,
        f"test_{clf.__name__}",
        create_classifier_function(clf)
    )

if __name__ == '__main__':
    unittest.main()

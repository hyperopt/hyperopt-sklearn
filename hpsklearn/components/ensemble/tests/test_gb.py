from .._gb import \
    gradient_boosting_classifier, \
    gradient_boosting_regression

import unittest
import numpy as np

from hyperopt import rand
from hpsklearn.estimator import hyperopt_estimator


class TestGradientBoostingClassification(unittest.TestCase):
    """
    Class for _gb classification testing
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

    def test_gb_classifier(self):
        """
        Instantiate gradient boosting classifier hyperopt estimator model
         fit and score model
        """
        model = hyperopt_estimator(
            regressor=gradient_boosting_classifier(name="classifier"),
            preprocessing=[],
            algo=rand.suggest,
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(self.X_train, self.Y_train)
        model.score(self.X_test, self.Y_test)

    test_gb_classifier.__name__ = f"test_{gradient_boosting_classifier.__name__}"


class TestGradientBoostingRegression(unittest.TestCase):
    """
    Class for _gb regression testing
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

    def test_gb_regressor(self):
        """
        Instantiate gradient boosting regressor hyperopt estimator model
         fit and score model
        """
        model = hyperopt_estimator(
            regressor=gradient_boosting_regression(name="regressor"),
            preprocessing=[],
            algo=rand.suggest,
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(self.X_train, self.Y_train)
        model.score(self.X_test, self.Y_test)

    test_gb_regressor.__name__ = f"test_{gradient_boosting_regression.__name__}"


if __name__ == '__main__':
    unittest.main()

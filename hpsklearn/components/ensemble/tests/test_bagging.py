from .._bagging import \
    bagging_classifier, \
    bagging_regressor

import unittest
import numpy as np

from hyperopt import rand
from hpsklearn.estimator import hyperopt_estimator


class TestBaggingClassification(unittest.TestCase):
    """
    Class for _bagging classification testing
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

    def test_bagging_classifier(self):
        """
        Instantiate bagging classifier hyperopt estimator model
         fit and score model
        """
        model = hyperopt_estimator(
            regressor=bagging_classifier(name="classifier"),
            preprocessing=[],
            algo=rand.suggest,
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(self.X_train, self.Y_train)
        model.score(self.X_test, self.Y_test)

    test_bagging_classifier.__name__ = f"test_{bagging_classifier.__name__}"


class TestBaggingRegression(unittest.TestCase):
    """
    Class for _bagging regression testing
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

    def test_bagging_regressor(self):
        """
        Instantiate bagging regressor hyperopt estimator model
         fit and score model
        """
        model = hyperopt_estimator(
            regressor=bagging_regressor(name="regressor"),
            preprocessing=[],
            algo=rand.suggest,
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(self.X_train, self.Y_train)
        model.score(self.X_test, self.Y_test)

    test_bagging_regressor.__name__ = f"test_{bagging_regressor.__name__}"


if __name__ == '__main__':
    unittest.main()

from .._weight_boosting import \
    ada_boost_classifier, \
    ada_boost_regressor

import unittest
import numpy as np

from hyperopt import rand
from hpsklearn.estimator import hyperopt_estimator


class TestWeightBoostingClassification(unittest.TestCase):
    """
    Class for _weight_boosting classification testing
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

    def test_adaboost_classifier(self):
        """
        Instantiate adaboost classifier hyperopt estimator model
         fit and score model
        """
        model = hyperopt_estimator(
            regressor=ada_boost_classifier(name="classifier"),
            preprocessing=[],
            algo=rand.suggest,
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(self.X_train, self.Y_train)
        model.score(self.X_test, self.Y_test)

    test_adaboost_classifier.__name__ = f"test_{ada_boost_classifier.__name__}"


class TestWeightBoostingRegression(unittest.TestCase):
    """
    Class for _weight_boosting regression testing
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

    def test_adaboost_regressor(self):
        """
        Instantiate adaboost regressor hyperopt estimator model
         fit and score model
        """
        model = hyperopt_estimator(
            regressor=ada_boost_regressor(name="regressor"),
            preprocessing=[],
            algo=rand.suggest,
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(self.X_train, self.Y_train)
        model.score(self.X_test, self.Y_test)

    test_adaboost_regressor.__name__ = f"test_{ada_boost_regressor.__name__}"


if __name__ == '__main__':
    unittest.main()

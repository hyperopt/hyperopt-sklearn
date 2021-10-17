import unittest

from hyperopt import rand

from hpsklearn.tests.utils import \
    StandardRegressorTest
from hpsklearn import hyperopt_estimator, \
    linear_regression


class TestBaseRegression(StandardRegressorTest):
    """
    Class for _base regression testing
    """
    def test_base_regressor(self):
        """
        Instantiate linear regressor hyperopt estimator model
         fit and score model
        """
        model = hyperopt_estimator(
            regressor=linear_regression(name="regressor"),
            preprocessing=[],
            algo=rand.suggest,
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(self.X_train, self.Y_train)
        model.score(self.X_test, self.Y_test)

    test_base_regressor.__name__ = f"test_{linear_regression.__name__}"


if __name__ == '__main__':
    unittest.main()

import unittest

from hyperopt import rand
from hpsklearn import HyperoptEstimator, isolation_forest
from sklearn.metrics import accuracy_score
from tests.utils import \
    IrisTest, \
    TrialsExceptionHandler


class TestIsolationForest(IrisTest):
    """
    Class for _iforest testing
    """
    @TrialsExceptionHandler
    def test_isolation_forest(self):
        """
        Instantiate isolation forest classifier hyperopt estimator model
         fit and score model
        """
        model = HyperoptEstimator(
            regressor=isolation_forest(name="i_forest"),
            preprocessing=[],
            algo=rand.suggest,
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(self.X_train, self.Y_train)
        accuracy_score(y_true=self.Y_test, y_pred=model.predict(self.X_test))

    test_isolation_forest.__name__ = f"test_{isolation_forest.__name__}"


if __name__ == '__main__':
    unittest.main()

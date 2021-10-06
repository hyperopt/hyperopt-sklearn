from .._iforest import isolation_forest

import unittest
import numpy as np

from hyperopt import rand
from hpsklearn.estimator import hyperopt_estimator
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class TestIsolationForest(unittest.TestCase):
    """
    Class for _iforest testing
    """
    def setUp(self):
        """
        Setup of iris dataset
        """
        rng = np.random.default_rng(seed=123)
        iris = load_iris()
        perm = rng.permutation(iris.target.size)
        iris.data = iris.data[perm]
        iris.target = iris.target[perm]

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            iris.data, iris.target
        )

    def test_isolation_forest(self):
        """
        Instantiate isolation forest classifier hyperopt estimator model
         fit and score model
        """
        model = hyperopt_estimator(
            regressor=isolation_forest(name="classifier"),
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

import unittest
import numpy as np

from hyperopt import rand

from hpsklearn import \
    hyperopt_estimator, \
    hist_gradient_boosting_classifier, \
    hist_gradient_boosting_regressor
from hpsklearn.tests.utils import \
    StandardClassifierTest, \
    StandardRegressorTest


class TestHistGradientBoostingClassification(StandardClassifierTest):
    """
    Class for _hist_gradient_boosting classification testing
    """

    def test_hist_gradient_boosting_classifier(self):
        """
        Instantiate hist gradient boosting classifier hyperopt estimator model
         fit and score model
        """
        model = hyperopt_estimator(
            classifier=hist_gradient_boosting_classifier(name="classifier"),
            preprocessing=[],
            algo=rand.suggest,
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(self.X_train, self.Y_train)
        model.score(self.X_test, self.Y_test)

    test_hist_gradient_boosting_classifier.__name__ = f"test_{hist_gradient_boosting_classifier.__name__}"


class TestHistGradientBoostingRegression(StandardRegressorTest):
    """
    Class for _hist_gradient_boosting regression testing
    """

    def test_hist_gradient_boosting_regressor(self):
        """
        Instantiate hist gradient boosting regressor hyperopt estimator model
         fit and score model
        """
        model = hyperopt_estimator(
            regressor=hist_gradient_boosting_regressor(name="regressor"),
            preprocessing=[],
            algo=rand.suggest,
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(self.X_train, self.Y_train)
        model.score(self.X_test, self.Y_test)

    test_hist_gradient_boosting_regressor.__name__ = f"test_{hist_gradient_boosting_regressor.__name__}"

    def test_poisson_function(self):
        """
        Instantiate hist gradient boosting hyperopt estimator model
         define 'criterion' = 'poisson'
         fit and score model
        """
        model = hyperopt_estimator(
            regressor=hist_gradient_boosting_regressor(name="poisson_regressor",
                                                       loss="poisson"),
            preprocessing=[],
            algo=rand.suggest,
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(np.abs(self.X_train), np.abs(self.Y_train))
        model.score(np.abs(self.X_test), np.abs(self.Y_test))

    test_poisson_function.__name__ = f"test_{hist_gradient_boosting_regressor.__name__}"


if __name__ == '__main__':
    unittest.main()

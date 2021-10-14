import unittest

from hyperopt import rand

from hpsklearn import \
    hyperopt_estimator, \
    gradient_boosting_classifier, \
    gradient_boosting_regression
from hpsklearn.tests.utils import \
    StandardClassifierTest, \
    StandardRegressorTest


class TestGradientBoostingClassification(StandardClassifierTest):
    """
    Class for _gb classification testing
    """
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


class TestGradientBoostingRegression(StandardRegressorTest):
    """
    Class for _gb regression testing
    """
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

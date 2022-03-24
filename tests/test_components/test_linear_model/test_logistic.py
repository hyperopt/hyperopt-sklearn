import unittest

from hpsklearn import \
    logistic_regression, \
    logistic_regression_cv
from tests.utils import \
    IrisTest, \
    TrialsExceptionHandler
from hyperopt import rand
from hpsklearn import HyperoptEstimator
from sklearn.metrics import accuracy_score


class TestLogisticRegression(IrisTest):
    """
    Class for _logistic regression testing
    """


def create_regression_attr(fn: callable):
    """
    Instantiate hyperopt estimator model

    Args:
        fn: estimator to test | callable

    fit and score model
    """
    @TrialsExceptionHandler
    def test_regressor(self):
        model = HyperoptEstimator(
            regressor=fn(name="regressor"),
            preprocessing=[],
            algo=rand.suggest,
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(self.X_train, self.Y_train)
        accuracy_score(y_true=self.Y_test, y_pred=model.predict(self.X_test))

    test_regressor.__name__ = f"test_{fn.__name__}"
    return test_regressor


for reg_fn in [logistic_regression, logistic_regression_cv]:
    setattr(
        TestLogisticRegression,
        f"test_{reg_fn.__name__}",
        create_regression_attr(reg_fn)
    )


if __name__ == '__main__':
    unittest.main()

import unittest

from hyperopt import rand
from hpsklearn import \
    hyperopt_estimator, \
    lars, \
    lasso_lars, \
    lars_cv, \
    lasso_lars_cv, \
    lasso_lars_ic
from hpsklearn.tests.utils import \
    StandardRegressorTest


class TestLeastAngleRegression(StandardRegressorTest):
    """
    Class for _least_angle regression testing
    """


# List of _least_angle regressors to test
regressors = [
    lars,
    lasso_lars,
    lars_cv,
    lasso_lars_cv,
    lasso_lars_ic
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


# Create unique _least_angle regression testing methods
#  with test_ prefix so that nose can see them
for reg in regressors:
    setattr(
        TestLeastAngleRegression,
        f"test_{reg.__name__}",
        create_regressor_function(reg)
    )


if __name__ == '__main__':
    unittest.main()

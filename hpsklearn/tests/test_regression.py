try:
    import unittest2 as unittest
except:
    import unittest

import numpy as np
from hyperopt import rand, tpe
from hpsklearn.estimator import hyperopt_estimator
from hpsklearn import components


class TestRegression(unittest.TestCase):

    def setUp(self):
        np.random.seed(123)
        self.X_train = np.random.randn(1000, 2)
        self.Y_train = self.X_train[:, 0] * 2
        self.X_test = np.random.randn(1000, 2)
        self.Y_test = self.X_test[:, 0] * 2

def create_function(reg_fn):
    def test_regressor(self):
        model = hyperopt_estimator(
            regressor=reg_fn('regressor'),
            preprocessing=[],
            algo=rand.suggest,
            trial_timeout=5.0,
            max_evals=5,
        )
        model.fit(self.X_train, self.Y_train)
        model.score(self.X_test, self.Y_test)

    test_regressor.__name__ = 'test_{0}'.format(reg_fn.__name__)
    return test_regressor


# List of regressors to test
regressors = [
    components.svr,
    components.svr_linear,
    components.svr_rbf,
    components.svr_poly,
    components.svr_sigmoid,
    components.knn_regression,
    components.ada_boost_regression,
    components.gradient_boosting_regression,
    components.random_forest_regression,
    components.extra_trees_regression,
    components.sgd_regression,
    components.lasso,
    components.elasticnet,
]


# Create unique methods with test_ prefix so that nose can see them
for reg in regressors:
    setattr(
        TestRegression,
        'test_{0}'.format(reg.__name__),
        create_function(reg)
    )

# Only test the xgboost regressor if the optional dependency is installed
try:
    import xgboost
except ImportError:
    xgboost = None

if xgboost is not None:
    setattr(
        TestRegression,
        'test_xgboost_regression',
        create_function(components.xgboost_regression)
    )

# Only test the lightgbm regressor if the optional dependency is installed
try:
    import lightgbm
except ImportError:
    lightgbm = None

if lightgbm is not None:
    setattr(
        TestRegression,
        'test_lightgmb_regression',
        create_function(components.lightgbm_regression)
    )

if __name__ == '__main__':
    unittest.main()

# -- flake8 eof

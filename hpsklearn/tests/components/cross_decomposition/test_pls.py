import unittest

from hpsklearn import \
    hyperopt_estimator, \
    cca, \
    pls_canonical, \
    pls_regression
from hpsklearn.tests.utils import \
    StandardRegressorTest, \
    generate_test_attributes
from hyperopt import rand


class TestPLSRegression(StandardRegressorTest):
    """
    Class for _pls regression testing
    """
    def test_cca(self):
        """
        Instantiate cca hyperopt estimator model
         fit and score model
        """
        model = hyperopt_estimator(
            regressor=cca(name="cca", n_components=1),
            preprocessing=[],
            algo=rand.suggest,
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(self.X_train, self.Y_train)
        model.score(self.X_test, self.Y_test)

    def test_pls_canonical(self):
        """
        Instantiate pls canonical hyperopt estimator model
         fit and score model
        """
        model = hyperopt_estimator(
            regressor=pls_canonical(name="pls_canonical", n_components=1),
            preprocessing=[],
            algo=rand.suggest,
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(self.X_train, self.Y_train)
        model.score(self.X_test, self.Y_test)


generate_test_attributes(
    TestClass=TestPLSRegression,
    fn_list=[pls_regression],
    is_classif=False,
)


if __name__ == '__main__':
    unittest.main()
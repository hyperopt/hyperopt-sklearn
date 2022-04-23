import unittest
import numpy as np

from hyperopt import rand

from hpsklearn import \
    HyperoptEstimator, \
    hist_gradient_boosting_classifier, \
    hist_gradient_boosting_regressor
from tests.utils import \
    StandardClassifierTest, \
    StandardRegressorTest, \
    generate_attributes, \
    TrialsExceptionHandler


class TestHistGradientBoostingClassification(StandardClassifierTest):
    """
    Class for _hist_gradient_boosting classification testing
    """


class TestHistGradientBoostingRegression(StandardRegressorTest):
    """
    Class for _hist_gradient_boosting regression testing
    """
    @TrialsExceptionHandler
    def test_poisson_function(self):
        """
        Instantiate hist gradient boosting hyperopt estimator model
         define 'criterion' = 'poisson'
         fit and score model
        """
        model = HyperoptEstimator(
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


generate_attributes(
    TestClass=TestHistGradientBoostingClassification,
    fn_list=[hist_gradient_boosting_classifier],
    is_classif=True
)


generate_attributes(
    TestClass=TestHistGradientBoostingRegression,
    fn_list=[hist_gradient_boosting_regressor],
    is_classif=False
)


if __name__ == '__main__':
    unittest.main()

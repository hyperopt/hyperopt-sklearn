import unittest
import numpy as np

from hyperopt import rand

from hpsklearn import \
    HyperoptEstimator, \
    random_forest_classifier, \
    random_forest_regressor, \
    extra_trees_classifier, \
    extra_trees_regressor
from tests.utils import \
    StandardClassifierTest, \
    StandardRegressorTest, \
    generate_attributes, \
    TrialsExceptionHandler


class TestForestClassification(StandardClassifierTest):
    """
    Class for _forest classification testing
    """


class TestForestRegression(StandardRegressorTest):
    """
    Class for _forest regression testing
    """
    @TrialsExceptionHandler
    def test_poisson_function(self):
        """
        Instantiate random forest regressor hyperopt estimator model
         define 'criterion' = 'poisson'
         fit and score model
        """
        model = HyperoptEstimator(
            regressor=random_forest_regressor(name="poisson_regressor",
                                              criterion="poisson"),
            preprocessing=[],
            algo=rand.suggest,
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(np.abs(self.X_train), np.abs(self.Y_train))
        model.score(np.abs(self.X_test), np.abs(self.Y_test))

    test_poisson_function.__name__ = f"test_{random_forest_regressor.__name__}"


generate_attributes(
    TestClass=TestForestClassification,
    fn_list=[random_forest_classifier, extra_trees_classifier],
    is_classif=True
)


generate_attributes(
    TestClass=TestForestRegression,
    fn_list=[random_forest_regressor, extra_trees_regressor],
    is_classif=False
)


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np

from hyperopt import rand

from hpsklearn import \
    HyperoptEstimator, \
    decision_tree_classifier, \
    decision_tree_regressor, \
    extra_tree_classifier, \
    extra_tree_regressor
from tests.utils import \
    StandardClassifierTest, \
    StandardRegressorTest, \
    generate_attributes, \
    TrialsExceptionHandler


class TestTreeClassification(StandardClassifierTest):
    """
    Class for tree._classes classification testing
    """


class TestTreeRegression(StandardRegressorTest):
    """
    Class for tree._classes regression testing
    """
    @TrialsExceptionHandler
    def test_poisson_function(self):
        """
        Instantiate decision tree regressor hyperopt estimator model
         define 'criterion' = 'poisson'
         fit and score model
        """
        model = HyperoptEstimator(
            regressor=decision_tree_regressor(name="poisson_regressor",
                                              criterion="poisson"),
            preprocessing=[],
            algo=rand.suggest,
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(np.abs(self.X_train), np.abs(self.Y_train))
        model.score(np.abs(self.X_test), np.abs(self.Y_test))

    test_poisson_function.__name__ = f"test_{decision_tree_regressor.__name__}"


generate_attributes(
    TestClass=TestTreeClassification,
    fn_list=[decision_tree_classifier, extra_tree_classifier],
    is_classif=True
)


generate_attributes(
    TestClass=TestTreeRegression,
    fn_list=[decision_tree_regressor, extra_tree_regressor],
    is_classif=False
)


if __name__ == '__main__':
    unittest.main()

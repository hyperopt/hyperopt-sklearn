import unittest

from hpsklearn import \
    HyperoptEstimator, \
    sgd_classifier, \
    sgd_regressor, \
    sgd_one_class_svm
from tests.utils import \
    StandardRegressorTest, \
    StandardClassifierTest, \
    generate_attributes, \
    TrialsExceptionHandler

from hyperopt import rand
from sklearn.metrics import accuracy_score


class TestSGDClassifier(StandardClassifierTest):
    """
    Class for _stochastic_gradient classification testing
    """


class TestSGDRegression(StandardRegressorTest):
    """
    Class for _stochastic_gradient regression testing
    """


class TestSGDOneClassSVM(StandardClassifierTest):
    """
    Class for SGDOneClassSVM testing
    """
    @TrialsExceptionHandler
    def test_sgd_one_class_svm(self):
        """
        Instantiate sgd one class svm classifier hyperopt estimator model
         fit and score model
        """
        model = HyperoptEstimator(
            regressor=sgd_one_class_svm(name="sgd_one_class_svm"),
            preprocessing=[],
            algo=rand.suggest,
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(self.X_train, self.Y_train)
        accuracy_score(y_true=self.Y_test, y_pred=model.predict(self.X_test))

    test_sgd_one_class_svm.__name__ = f"test_{sgd_one_class_svm.__name__}"


generate_attributes(
    TestClass=TestSGDClassifier,
    fn_list=[sgd_classifier],
    is_classif=True,
)


generate_attributes(
    TestClass=TestSGDRegression,
    fn_list=[sgd_regressor],
    is_classif=False,
)


if __name__ == '__main__':
    unittest.main()

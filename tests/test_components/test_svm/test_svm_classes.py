import unittest

from hyperopt import rand
from hpsklearn import \
    HyperoptEstimator, \
    linear_svc, \
    linear_svr, \
    nu_svc, \
    nu_svr, \
    one_class_svm, \
    svc, \
    svr
from tests.utils import \
    StandardRegressorTest, \
    StandardClassifierTest, \
    IrisTest, \
    generate_attributes, \
    TrialsExceptionHandler
from sklearn.metrics import accuracy_score


class TestSVMClassifier(StandardClassifierTest):
    """
    Class for _classes classification testing
    """


class TestSVMRegression(StandardRegressorTest):
    """
    Class for _classes regression testing
    """


class TestOneClassSVM(IrisTest):
    """
    Class for one_class_svm testing
    """
    @TrialsExceptionHandler
    def test_one_class_svm(self):
        """
        Instantiate one_class_svm hyperopt estimator model
         fit and score model
        """
        model = HyperoptEstimator(
            regressor=one_class_svm(name="one_class_svm_regressor"),
            preprocessing=[],
            algo=rand.suggest,
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(self.X_train, self.Y_train)
        accuracy_score(y_true=self.Y_test, y_pred=model.predict(self.X_test))

    test_one_class_svm.__name__ = f"test_{one_class_svm.__name__}"


generate_attributes(
    TestClass=TestSVMClassifier,
    fn_list=[linear_svc, nu_svc, svc],
    is_classif=True,
)


generate_attributes(
    TestClass=TestSVMRegression,
    fn_list=[linear_svr, nu_svr, svr],
    is_classif=False,
)


if __name__ == '__main__':
    unittest.main()

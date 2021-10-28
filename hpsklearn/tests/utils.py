import typing
import unittest
import numpy as np

from hyperopt import rand
from hpsklearn import hyperopt_estimator


class StandardClassifierTest(unittest.TestCase):
    """
    Standard class for classification testing
    """

    def setUp(self):
        """
        Setup of randomly generated data
        """
        np.random.seed(123)
        self.X_train = np.random.randn(1000, 2)
        self.Y_train = (self.X_train[:, 0] > 0).astype("int")
        self.X_test = np.random.randn(1000, 2)
        self.Y_test = (self.X_test[:, 0] > 0).astype("int")


class StandardRegressorTest(unittest.TestCase):
    """
    Standard class for regressor testing
    """

    def setUp(self):
        """
        Setup of randomly generated data
        """
        np.random.seed(123)
        self.X_train = np.random.randn(1000, 2)
        self.Y_train = self.X_train[:, 0] * 2
        self.X_test = np.random.randn(1000, 2)
        self.Y_test = self.X_test[:, 0] * 2


class MultiTaskRegressorTest(unittest.TestCase):
    """
    Standard class for multi task regressor testing
    """

    def setUp(self):
        """
        Setup of randomly generated data
        """
        np.random.seed(123)
        self.X_train = np.random.randn(1000, 2)
        self.Y_train = self.X_train * 2
        self.X_test = np.random.randn(1000, 2)
        self.Y_test = self.X_test * 2


class StandardPreprocessingTest(StandardClassifierTest):
    """
    Standard class for preprocessing testing

    The required data is the same as the data
     required for standard classification testing.
    """


def create_function(fn: callable, is_classif: bool, non_negative_input: bool, non_negative_output: bool):
    """
    Instantiate standard hyperopt estimator model

    Args:
        fn: estimator to test | callable
        is_classif: estimator is classifier | bool
        non_negative_input: estimator input non negative | bool
        non_negative_output: estimator output non negative | bool

     fit and score model
    """

    def test_estimator(self):
        if is_classif:
            model = hyperopt_estimator(
                classifier=fn("classifier"),
                preprocessing=[],
                algo=rand.suggest,
                trial_timeout=10.0,
                max_evals=5,
            )
        else:
            model = hyperopt_estimator(
                regressor=fn("regressor"),
                preprocessing=[],
                algo=rand.suggest,
                trial_timeout=10.0,
                max_evals=5,
            )
        model.fit(np.abs(self.X_train) if non_negative_input else self.X_train,
                  np.abs(self.Y_train) if non_negative_output else self.Y_train)
        model.score(np.abs(self.X_test) if non_negative_input else self.X_test,
                    np.abs(self.Y_test) if non_negative_output else self.Y_test)

    test_estimator.__name__ = f"test_{fn.__name__}"
    return test_estimator


def generate_test_attributes(TestClass,
                             fn_list: list[callable],
                             is_classif: bool,
                             non_negative_input: bool = False,
                             non_negative_output: bool = False):
    """
    Generate class methods

    Args:
        TestClass: main test class | unittest.TestCase
        fn_list: list of estimators to test | list[callable]
        is_classif: estimator is classifier | bool
        non_negative_input: estimator input non negative | bool
        non_negative_output: estimator output non negative | bool
    """
    # Create unique testing methods
    #  with test_ prefix so that nose can see them
    for fn in fn_list:
        setattr(
            TestClass,
            f"test_{fn.__name__}",
            create_function(fn=fn,
                            is_classif=is_classif,
                            non_negative_input=non_negative_input,
                            non_negative_output=non_negative_output)
        )

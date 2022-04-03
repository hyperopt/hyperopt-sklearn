import unittest
import numpy as np

from hyperopt import rand
from hyperopt.exceptions import AllTrialsFailed
from hpsklearn import HyperoptEstimator

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import typing


def TrialsExceptionHandler(fn):
    """
    Decorator for handling exceptions in trials
    """
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except AllTrialsFailed:
            print("\n---\nAllTrialsFailed was raised, np.isnan(t['result']['loss']) is True.\n---\n")
    return wrapper


def RetryOnTrialsException(fn):
    """
    Decorator for retrying tests that fail due to AllTrialsFailed
    """
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except AllTrialsFailed:
            print("\n---\nAllTrialsFailed was raised, np.isnan(t['result']['loss']) is True.\n---\n")
            return wrapper(*args, **kwargs)
    return wrapper


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


class IrisTest(unittest.TestCase):
    """
    Standard class for testing with iris dataset
    """

    def setUp(self):
        """
        Setup of iris dataset
        """
        rng = np.random.default_rng(seed=123)
        iris = load_iris()
        perm = rng.permutation(iris.target.size)
        iris.data = iris.data[perm]
        iris.target = iris.target[perm]

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            iris.data, iris.target
        )


class StandardPreprocessingTest(StandardClassifierTest):
    """
    Standard class for preprocessing testing

    The required data is the same as the data
     required for standard classification testing.
    """


def create_function(fn: callable,
                    is_classif: bool,
                    non_negative_input: bool,
                    non_negative_output: bool,
                    trial_timeout: float,
                    max_evals: int):
    """
    Instantiate standard hyperopt estimator model

    Args:
        fn: estimator to test | callable
        is_classif: estimator is classifier | bool
        non_negative_input: estimator input non negative | bool
        non_negative_output: estimator output non negative | bool
        trial_timeout: kill trial evaluations after this many seconds | float
        max_evals: evaluate up to this-many configurations | int

     fit and score model
    """

    @TrialsExceptionHandler
    def test_estimator(self):
        if is_classif:
            model = HyperoptEstimator(
                classifier=fn("classifier"),
                preprocessing=[],
                algo=rand.suggest,
                trial_timeout=trial_timeout,
                max_evals=max_evals,
            )
        else:
            model = HyperoptEstimator(
                regressor=fn("regressor"),
                preprocessing=[],
                algo=rand.suggest,
                trial_timeout=trial_timeout,
                max_evals=max_evals,
            )
        model.fit(np.abs(self.X_train) if non_negative_input else self.X_train,
                  np.abs(self.Y_train) if non_negative_output else self.Y_train)
        model.score(np.abs(self.X_test) if non_negative_input else self.X_test,
                    np.abs(self.Y_test) if non_negative_output else self.Y_test)

    test_estimator.__name__ = f"test_{fn.__name__}"
    return test_estimator


def create_preprocessing_function(pre_fn, classifier):
    """
    Instantiate gaussian_nb hyperopt estimator model
     'pre_fn' regards the preprocessor
     fit and score model
    """

    @TrialsExceptionHandler
    def test_preprocessor(self):
        model = HyperoptEstimator(
            classifier=classifier("classifier"),
            preprocessing=[pre_fn("preprocessing")],
            algo=rand.suggest,
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(self.X_train, self.Y_train)
        model.score(self.X_test, self.Y_test)

    test_preprocessor.__name__ = f"test_{pre_fn.__name__}"
    return test_preprocessor


# Create unique _data preprocessing algorithms
#  with test_ prefix so that unittest can see them
def generate_preprocessor_attributes(TestClass,
                                     preprocessor_list: typing.List[callable],
                                     classifier: callable):
    """
    Generate class methods

    Args:
        TestClass: main test class | unittest.TestCase
        preprocessor_list: list of preprocessors to test | list[callable]
        classifier: classifier to test preprocessors with | callable
    """
    for pre in preprocessor_list:
        setattr(
            TestClass,
            f"test_{pre.__name__}",
            create_preprocessing_function(pre, classifier)
        )


def generate_attributes(TestClass,
                        fn_list: typing.List[callable],
                        is_classif: bool,
                        non_negative_input: bool = False,
                        non_negative_output: bool = False,
                        trial_timeout: float = 10.0,
                        max_evals: int = 5):
    """
    Generate class methods

    Args:
        TestClass: main test class | unittest.TestCase
        fn_list: list of estimators to test | list[callable]
        is_classif: estimator is classifier | bool
        non_negative_input: estimator input non negative | bool
        non_negative_output: estimator output non negative | bool
        trial_timeout: kill trial evaluations after this many seconds | float
        max_evals: evaluate up to this-many configurations | int
    """
    # Create unique testing methods
    #  with test_ prefix so that unittest can see them
    for fn in fn_list:
        setattr(
            TestClass,
            f"test_{fn.__name__}",
            create_function(fn=fn,
                            is_classif=is_classif,
                            non_negative_input=non_negative_input,
                            non_negative_output=non_negative_output,
                            trial_timeout=trial_timeout,
                            max_evals=max_evals)
        )

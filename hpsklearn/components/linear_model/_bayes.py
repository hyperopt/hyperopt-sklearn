from hpsklearn.components._base import validate

from hyperopt.pyll import scope
from hyperopt import hp

from sklearn import linear_model
import numpy as np


@scope.define
def sklearn_BayesianRidge(*args, **kwargs):
    return linear_model.BayesianRidge(*args, **kwargs)


@scope.define
def sklearn_ARDRegression(*args, **kwargs):
    return linear_model.ARDRegression(*args, **kwargs)


def _bayes_n_iter(name: str):
    """
    Declaration search space 'n_iter' parameter
    """
    return scope.int(hp.qloguniform(name, low=np.log(150), high=np.log(450), q=1.0))


def _bayes_tol(name: str):
    """
    Declaration search space 'tol' parameter
    """
    return hp.lognormal(name, mu=np.log(1e-3), sigma=np.log(10))


def _bayes_alpha_lambda(name: str):
    """
    Declaration search space 'alpha_1', 'alpha_2',
     'lambda_1' and 'lambda_2' parameters
    """
    return hp.lognormal(name, mu=np.log(1e-6), sigma=np.log(10))


@validate(params=["n_iter"],
          validation_test=lambda param: isinstance(param, int) and param > 1,
          msg="Invalid parameter '%s' with value '%s'. Parameter value must exceed 1.")
@validate(params=["alpha_1", "alpha_2", "lambda_1", "lambda_2"],
          validation_test=lambda param: isinstance(param, float) and param >= 0,
          msg="Invalid parameter '%s' with value '%s'. Parameter value must be equal to or exceed 0.")
def _bayes_hp_space(
        name_func,
        n_iter: int = None,
        tol: float = None,
        alpha_1: float = None,
        alpha_2: float = None,
        lambda_1: float = None,
        lambda_2: float = None,
        compute_score: bool = False,
        fit_intercept: bool = True,
        copy_X: bool = True,
        verbose: bool = False,
):
    """
    Hyper parameter search space for
     bayesian ridge
     ard regression
    """
    hp_space = dict(
        n_iter=n_iter or _bayes_n_iter(name_func("n_iter")),
        tol=_bayes_tol(name_func("tol")) if tol is None else tol,
        alpha_1=alpha_1 or _bayes_alpha_lambda(name_func("alpha_1")),
        alpha_2=alpha_2 or _bayes_alpha_lambda(name_func("alpha_2")),
        lambda_1=lambda_1 or _bayes_alpha_lambda(name_func("lambda_1")),
        lambda_2=lambda_2 or _bayes_alpha_lambda(name_func("lambda_2")),
        compute_score=compute_score,
        fit_intercept=fit_intercept,
        copy_X=copy_X,
        verbose=verbose,
    )
    return hp_space


def bayesian_ridge(name: str,
                   alpha_init: float = None,
                   lambda_init: float = None,
                   **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.BayesianRidge model.

    Args:
        name: name | str
        alpha_init: init precision of noise | float
        lambda_init: init precision of weights | float

    See help(hpsklearn.components.linear_model._bayes._bayes_hp_space)
    for info on additional available bayes arguments.
    """
    def _name(msg):
        return f"{name}.bayesian_ridge_{msg}"

    hp_space = _bayes_hp_space(_name, **kwargs)
    hp_space["alpha_init"] = alpha_init
    hp_space["lambda_init"] = lambda_init

    return scope.sklearn_BayesianRidge(**hp_space)


def ard_regression(name: str, threshold_lambda: float = 10000, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.ARDRegression model.

    Args:
        name: name | str
        threshold_lambda: threshold for removing weights | float

    See help(hpsklearn.components.linear_model._bayes._bayes_hp_space)
    for info on additional available bayes arguments.
    """
    def _name(msg):
        return f"{name}.ard_regression_{msg}"

    hp_space = _bayes_hp_space(_name, **kwargs)
    hp_space["threshold_lambda"] = threshold_lambda

    return scope.sklearn_ARDRegression(**hp_space)

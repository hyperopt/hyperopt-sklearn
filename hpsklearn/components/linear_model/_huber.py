from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import linear_model
import numpy as np
import typing


@scope.define
def sklearn_HuberRegressor(*args, **kwargs):
    return linear_model.HuberRegressor(*args, **kwargs)


def _glm_epsilon(name: str):
    """
    Declaration search space 'epsilon' parameter
    """
    return hp.normal(name, mu=1.35, sigma=0.05)


def _glm_max_iter(name: str):
    """
    Declaration search space 'max_iter' parameter
    """
    return scope.int(hp.uniform(name, low=80, high=120))


def _glm_alpha(name: str):
    """
    Declaration search space 'alpha' parameter
    """
    return hp.loguniform(name, low=np.log(1e-5), high=np.log(1e-2))


def _glm_tol(name: str):
    """
    Declaration search space 'tol' parameter
    """
    return hp.loguniform(name, low=np.log(1e-6), high=np.log(1e-4))


@validate(params=["epsilon", "max_iter"],
          validation_test=lambda param: not isinstance(param, float) or param > 1,
          msg="Invalid parameter '%s' with value '%s'. Parameter must exceed 1.")
def huber_regressor(name: str,
                    epsilon: typing.Union[float, Apply] = None,
                    max_iter: typing.Union[int, Apply] = None,
                    alpha: typing.Union[float, Apply] = None,
                    warm_start: bool = False,
                    fit_intercept: bool = True,
                    tol: typing.Union[float, Apply] = None,
                    **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.HuberRegressor model.

    Args:
        name: name | str
        epsilon: samples classified as outliers | float
        max_iter: max number of iterations | int
        alpha: regularization parameter | float
        warm_start: reuse attributes | bool
        fit_intercept: whether to fit the intercept | bool
        tol: threshold to stop iteration | float
    """
    def _name(msg):
        return f"{name}.huber_regressor_{msg}"

    hp_space = dict(
        epsilon=_glm_epsilon(_name("epsilon")) if tol is None else epsilon,
        max_iter=_glm_max_iter(_name("max_iter")) if tol is None else max_iter,
        alpha=_glm_alpha(_name("alpha")) if tol is None else alpha,
        warm_start=warm_start,
        fit_intercept=fit_intercept,
        tol=_glm_tol(_name("tol")) if tol is None else tol,
        **kwargs
    )

    return scope.sklearn_HuberRegressor(**hp_space)

from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import linear_model
import numpy as np
import typing


@scope.define
def sklearn_PoissonRegressor(*args, **kwargs):
    return linear_model.PoissonRegressor(*args, **kwargs)


@scope.define
def sklearn_GammaRegressor(*args, **kwargs):
    return linear_model.GammaRegressor(*args, **kwargs)


@scope.define
def sklearn_TweedieRegressor(*args, **kwargs):
    return linear_model.TweedieRegressor(*args, **kwargs)


def _glm_max_iter(name: str):
    """
    Declaration search space 'max_iter' parameter
    """
    return scope.int(hp.uniform(name, low=80, high=120))


def _glm_tol(name: str):
    """
    Declaration search space 'tol' parameter
    """
    return hp.loguniform(name, low=np.log(1e-5), high=np.log(1e-2))


def _glm_power(name: str):
    """
    Declaration search space 'power' parameter
    """
    return hp.choice(name, [0, 1, 2, 3])


@validate(params=["max_iter"],
          validation_test=lambda param: not isinstance(param, int) or param > 1,
          msg="Invalid parameter '%s' with value '%s'. Parameter must exceed 1.")
def _glm_hp_space(
        name_func,
        alpha: float = 1,
        fit_intercept: bool = True,
        max_iter: typing.Union[int, Apply] = 100,
        tol: typing.Union[float, Apply] = 1e-4,
        warm_start: bool = False,
        verbose: int = 0,
        **kwargs
):
    """
    Hyper parameter search space for
     poisson regressor
     gamma regressor
     tweedie regressor
    """
    hp_space = dict(
        alpha=alpha,
        fit_intercept=fit_intercept,
        max_iter=_glm_max_iter(name_func("max_iter")) if max_iter is None else max_iter,
        tol=_glm_tol(name_func("tol")) if tol is None else tol,
        warm_start=warm_start,
        verbose=verbose,
        **kwargs
    )
    return hp_space


def poisson_regressor(name: str, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.PoissonRegressor model.

    Args:
        name: name | str

    See help(hpsklearn.components.linear_model._glm._glm_hp_space)
    for info on additional available glm arguments.
    """

    def _name(msg):
        return f"{name}.poisson_regressor_{msg}"

    hp_space = _glm_hp_space(_name, **kwargs)

    return scope.sklearn_PoissonRegressor(**hp_space)


def gamma_regressor(name: str, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.GammaRegressor model.

    Args:
        name: name | str

    See help(hpsklearn.components.linear_model._glm._glm_hp_space)
    for info on additional available glm arguments.
    """

    def _name(msg):
        return f"{name}.gamma_regressor_{msg}"

    hp_space = _glm_hp_space(_name, **kwargs)

    return scope.sklearn_GammaRegressor(**hp_space)


@validate(params=["link"],
          validation_test=lambda param: not isinstance(param, str) or param in ["auto", "identity", "log"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['auto', 'identity', 'log'].")
def tweedie_regressor(name: str,
                      power: typing.Union[float, Apply] = None,
                      link: str = "auto",
                      **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.TweedieRegressor model.

    Args:
        name: name | str
        power: target distribution | float
        link: link function of GLM | str

    See help(hpsklearn.components.linear_model._glm._glm_hp_space)
    for info on additional available glm arguments.
    """

    def _name(msg):
        return f"{name}.tweedie_regressor_{msg}"

    hp_space = _glm_hp_space(_name, **kwargs)
    hp_space["power"] = _glm_power(_name("power")) if power is None else power
    hp_space["link"] = link

    return scope.sklearn_TweedieRegressor(**hp_space)

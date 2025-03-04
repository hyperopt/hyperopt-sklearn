from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import linear_model
import numpy as np
import typing

from collections.abc import Iterable


@scope.define
def sklearn_Ridge(*args, **kwargs):
    return linear_model.Ridge(*args, **kwargs)


@scope.define
def sklearn_RidgeCV(*args, **kwargs):
    return linear_model.RidgeCV(*args, **kwargs)


@scope.define
def sklearn_RidgeClassifier(*args, **kwargs):
    return linear_model.RidgeClassifier(*args, **kwargs)


@scope.define
def sklearn_RidgeClassifierCV(*args, **kwargs):
    return linear_model.RidgeClassifierCV(*args, **kwargs)


def _ridge_alpha(name: str):
    """
    Declaration search space 'alpha' parameter
    """
    return hp.loguniform(name, np.log(1e-3), np.log(1e3))


def _ridge_max_iter(name: str):
    """
    Declaration search space 'max_iter' parameter
    """
    return scope.int(hp.uniform(name, 750, 1250))


def _ridge_tol(name: str):
    """
    Declaration search space 'tol' parameter
    """
    return hp.loguniform(name, np.log(1e-5), np.log(1e-2))


def _ridge_cv(name: str):
    """
    Declaration search space 'cv' parameter
    """
    return hp.pchoice(name, [
        # create custom distribution
        (0.0625, 3),
        (0.175, 4),
        (0.525, 5),
        (0.175, 6),
        (0.0625, 7),
    ])


def _ridge_alphas(name: str):
    """
    Declaration search space 'alphas' parameter
    """
    return hp.choice(name, [
        (0.01, 0.1, 1.0),
        (0.1, 1.0, 10.0),
        (1.0, 10.0, 100.0)
    ])


@validate(params=["solver"],
          validation_test=lambda param: not isinstance(param, str) or param in
                                        ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"],  # noqa
          msg="Invalid parameter '%s' with value '%s'. "
              "Value must be in ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'].")
@validate(params=["alpha"],
          validation_test=lambda param: not isinstance(param, int) or param >= 0,
          msg="Invalid parameter '%s' with value '%s'. Alpha must be positive.")
def _ridge_hp_space(
        name_func,
        alpha: typing.Union[float, np.ndarray, Apply] = None,
        fit_intercept: bool = True,
        copy_X: bool = True,
        max_iter: typing.Union[int, Apply] = None,
        tol: typing.Union[float, Apply] = None,
        solver: typing.Union[str, Apply] = "auto",
        positive: bool = False,
        random_state=None,
        **kwargs,
):
    """
    Hyper parameter search space for
     ridge
     ridge classifier
    """
    hp_space = dict(
        alpha=_ridge_alpha(name_func("alpha")) if alpha is None else alpha,
        fit_intercept=fit_intercept,
        copy_X=copy_X,
        max_iter=_ridge_max_iter(name_func("max_iter")) if max_iter is None else max_iter,
        tol=_ridge_tol(name_func("tol")) if tol is None else tol,
        solver=solver,
        positive=positive,
        random_state=random_state,
        **kwargs
    )
    return hp_space


def _ridge_cv_hp_space(
        name_func,
        alphas: typing.Union[np.ndarray, Apply] = None,
        fit_intercept: bool = True,
        scoring: typing.Union[str, callable] = None,
        cv: typing.Union[int, Iterable, typing.Generator, Apply] = None,
        **kwargs
):
    """
    Hyper parameter search space for
     ridge cv
     ridge classifier cv
    """
    hp_space = dict(
        alphas=_ridge_alphas(name_func("alphas")) if alphas is None else alphas,
        fit_intercept=fit_intercept,
        scoring=scoring,
        cv=_ridge_cv(name_func("cv")) if cv is None else cv,
        **kwargs
    )
    return hp_space


def ridge(name: str, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.Ridge model.

    Args:
        name: name | str

    See help(hpsklearn.components.linear_model._ridge._ridge_hp_space)
    for info on additional available ridge arguments.
    """

    def _name(msg):
        return f"{name}.ridge_{msg}"

    hp_space = _ridge_hp_space(_name, **kwargs)

    return scope.sklearn_Ridge(**hp_space)


@validate(params=["gcv_mode"],
          validation_test=lambda param: not isinstance(param, str) or param in ["auto", "svd", "eigen"],
          msg="Invalid parameter '%s' with value '%s'. Value must be one of 'auto', 'svd', 'eigen'.")
def ridge_cv(name: str,
             gcv_mode: typing.Union[str, Apply] = "auto",
             alpha_per_target: bool = False,
             store_cv_results: bool = False,
             **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.RidgeCV model.

    Args:
        name: name | str
        gcv_mode: strategy for cv | str
        alpha_per_target: optimize alpha per target | bool

    See help(hpsklearn.components.linear_model._ridge._ridge_hp_space)
    for info on additional available ridge arguments.
    """

    def _name(msg):
        return f"{name}.ridge_cv_{msg}"

    hp_space = _ridge_cv_hp_space(_name, **kwargs)
    hp_space["gcv_mode"] = gcv_mode
    hp_space["alpha_per_target"] = alpha_per_target
    hp_space["store_cv_results"] = store_cv_results

    return scope.sklearn_RidgeCV(**hp_space)


@validate(params=["class_weight"],
          validation_test=lambda param: not isinstance(param, str) or param == "balanced",
          msg="Invalid parameter '%s' with value '%s'. Value must be 'balanced'")
def ridge_classifier(name: str, class_weight: typing.Union[dict, str] = None, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.RidgeClassifier model.

    Args:
        name: name | str
        class_weight: class_weight fit parameters | dict or str

    See help(hpsklearn.components.linear_model._ridge._ridge_hp_space)
    for info on additional available ridge arguments.
    """

    def _name(msg):
        return f"{name}.ridge_classifier_{msg}"

    hp_space = _ridge_hp_space(_name, **kwargs)
    hp_space["class_weight"] = class_weight

    return scope.sklearn_RidgeClassifier(**hp_space)


@validate(params=["class_weight"],
          validation_test=lambda param: not isinstance(param, str) or param == "balanced",
          msg="Invalid parameter '%s' with value '%s'. Value must be 'balanced'")
def ridge_classifier_cv(name: str,
                        class_weight: typing.Union[dict, str] = None,
                        store_cv_results: bool = False,
                        **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.RidgeClassifierCV model.

    Args:
        name: name | str
        class_weight: class_weight fit parameters | dict or str

    See help(hpsklearn.components.linear_model._ridge._ridge_hp_space)
    for info on additional available ridge arguments.
    """

    def _name(msg):
        return f"{name}.ridge_classifier_cv_{msg}"

    hp_space = _ridge_cv_hp_space(_name, **kwargs)
    hp_space["class_weight"] = class_weight
    hp_space["store_cv_results"] = store_cv_results

    return scope.sklearn_RidgeClassifierCV(**hp_space)

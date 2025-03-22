from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

import numpy as np
import typing

try:
    import xgboost
except ImportError:
    xgboost = None


@scope.define
def sklearn_XGBClassifier(*args, **kwargs):
    if xgboost is None:
        raise ImportError("No module named xgboost")
    return xgboost.XGBClassifier(*args, **kwargs)


@scope.define
def sklearn_XGBRegressor(*args, **kwargs):
    if xgboost is None:
        raise ImportError("No module named xgboost")
    return xgboost.XGBRegressor(*args, **kwargs)


def _xgboost_max_depth(name: str):
    """
    Declaration search space 'max_depth' parameter
    """
    return scope.int(hp.uniform(name, 1, 11))


def _xgboost_learning_rate(name: str):
    """
    Declaration search space 'learning_rate' parameter
    """
    return hp.loguniform(name, np.log(0.0001), np.log(0.5)) - 0.0001


def _xgboost_n_estimators(name: str):
    """
    Declaration search space 'n_estimators' parameter
    """
    return scope.int(hp.quniform(name, 100, 6000, 200))


def _xgboost_gamma(name: str):
    """
    Declaration search space 'gamma' parameter
    """
    return hp.loguniform(name, np.log(0.0001), np.log(5)) - 0.0001


def _xgboost_min_child_weight(name: str):
    """
    Declaration search space 'min_child_weight' parameter
    """
    return scope.int(hp.loguniform(name, np.log(1), np.log(100)))


def _xgboost_subsample(name: str):
    """
    Declaration search space 'subsample' parameter
    """
    return hp.uniform(name, 0.5, 1)


def _xgboost_colsample_bytree(name: str):
    """
    Declaration search space 'colsample_bytree' parameter
    """
    return hp.uniform(name, 0.5, 1)


def _xgboost_colsample_bylevel(name: str):
    """
    Declaration search space 'colsample_bylevel' parameter
    """
    return hp.uniform(name, 0.5, 1)


def _xgboost_reg_alpha(name: str):
    """
    Declaration search space 'reg_alpha' parameter
    """
    return hp.loguniform(name, np.log(0.0001), np.log(1)) - 0.0001


def _xgboost_reg_lambda(name: str):
    """
    Declaration search space 'reg_lambda' parameter
    """
    return hp.loguniform(name, np.log(1), np.log(4))


def _xgboost_random_state(name: str):
    """
    Declaration search space 'random_state' parameter
    """
    return hp.randint(name, 5)


def _xgboost_hp_space(
        name_func,
        max_depth: typing.Union[int, Apply] = None,
        learning_rate: typing.Union[float, Apply] = None,
        n_estimators: typing.Union[int, Apply] = None,
        gamma: typing.Union[float, Apply] = None,
        min_child_weight: typing.Union[float, Apply] = None,
        max_delta_step: float = 0,
        subsample: typing.Union[float, Apply] = None,
        colsample_bytree: typing.Union[float, Apply] = None,
        colsample_bylevel: typing.Union[float, Apply] = None,
        reg_alpha: typing.Union[float, Apply] = None,
        reg_lambda: typing.Union[float, Apply] = None,
        scale_pos_weight: float = 1,
        base_score: float = 0.5,
        random_state=None,
        n_jobs: int = -1,
        **kwargs):
    """
    Hyper parameter search space for
     xgboost classifier
     xgboost regressor
    """
    hp_space = dict(
        max_depth=_xgboost_max_depth(name_func("max_depth")) if max_depth is None else max_depth,
        learning_rate=_xgboost_learning_rate(name_func("learning_rate")) if learning_rate is None else learning_rate,
        n_estimators=_xgboost_n_estimators(name_func("n_estimators")) if n_estimators is None else n_estimators,
        gamma=_xgboost_gamma(name_func("gamma")) if gamma is None else gamma,
        min_child_weight=_xgboost_min_child_weight(name_func("min_child_weight"))
        if min_child_weight is None else min_child_weight,
        max_delta_step=max_delta_step,
        subsample=_xgboost_subsample(name_func("subsample")) if subsample is None else subsample,
        colsample_bytree=_xgboost_colsample_bytree(name_func("colsample_bytree"))
        if colsample_bytree is None else colsample_bytree,
        colsample_bylevel=_xgboost_colsample_bylevel(name_func("colsample_bylevel"))
        if colsample_bylevel is None else colsample_bylevel,
        reg_alpha=_xgboost_reg_alpha(name_func("reg_alpha")) if reg_alpha is None else reg_alpha,
        reg_lambda=_xgboost_reg_lambda(name_func("reg_lambda")) if reg_lambda is None else reg_lambda,
        scale_pos_weight=scale_pos_weight,
        base_score=base_score,
        seed=_xgboost_random_state(name_func("random_state")) if random_state is None else random_state,
        n_jobs=n_jobs,
        **kwargs
    )
    return hp_space


@validate(params=["objective"],
          validation_test=lambda param: not isinstance(param, str) or param in ["binary:logistic", "binary:logitraw"],
          msg="Invalid parameter '%s' with value '%s'. Value must be 'binary:logistic' or 'binary:logitraw'.")
def xgboost_classification(name: str, objective: typing.Union[str, Apply] = None, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a xgboost.XGBClassifier model.

    Args:
        name: name | str
        objective: 'binary:logistic' or 'binary:logitraw' | str

    See help(hpsklearn.components._xgboost_hp_space) for info on
    additional available XGBoost arguments.
    """

    def _name(msg):
        return f"{name}.xgboost_clf_{msg}"

    hp_space = _xgboost_hp_space(_name, **kwargs)
    hp_space["objective"] = "binary:logistic" if objective is None else objective

    return scope.sklearn_XGBClassifier(**hp_space)


@validate(params=["objective"],
          validation_test=lambda param: not isinstance(param, str) or param in ["reg:squarederror", "count:poisson"],
          msg="Invalid parameter '%s' with value '%s'. Value must be 'reg:squarederror' or 'count:poisson'.")
def xgboost_regression(name: str, objective: typing.Union[str, Apply] = None, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a xgboost.XGBRegressor model.

    Args:
        name: name | str
        objective: 'reg:squarederror' or 'count:poisson' | str

    See help(hpsklearn.components._xgboost_hp_space) for info on
    additional available XGBoost arguments.
    """

    def _name(msg):
        return f"{name}.xgboost_reg_{msg}"

    hp_space = _xgboost_hp_space(_name, **kwargs)
    hp_space["objective"] = "reg:squarederror" if objective is None else objective

    return scope.sklearn_XGBRegressor(**hp_space)

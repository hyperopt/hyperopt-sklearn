from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

import numpy as np
import typing

try:
    import lightgbm
except ImportError:
    lightgbm = None


@scope.define
def sklearn_LGBMClassifier(*args, **kwargs):
    if lightgbm is None:
        raise ImportError("No module named lightgbm")
    return lightgbm.LGBMClassifier(*args, **kwargs)


@scope.define
def sklearn_LGBMRegressor(*args, **kwargs):
    if lightgbm is None:
        raise ImportError("No module named lightgbm")
    return lightgbm.LGBMRegressor(*args, **kwargs)


def _lightgbm_max_depth(name):
    """
    Declaration search space 'max_depth' parameter
    """
    return scope.int(hp.uniform(name, 1, 11))


def _lightgbm_num_leaves(name):
    """
    Declaration search space 'num_leaves' parameter
    """
    return scope.int(hp.uniform(name, 2, 121))


def _lightgbm_learning_rate(name):
    """
    Declaration search space 'learning_rate' parameter
    """
    return hp.loguniform(name, np.log(0.0001), np.log(0.5)) - 0.0001


def _lightgbm_n_estimators(name):
    """
    Declaration search space 'n_estimators' parameter
    """
    return scope.int(hp.quniform(name, 100, 6000, 200))


def _lightgbm_min_child_weight(name):
    """
    Declaration search space 'min_child_weight' parameter
    """
    return scope.int(hp.loguniform(name, np.log(1), np.log(100)))


def _lightgbm_subsample(name):
    """
    Declaration search space 'subsample' parameter
    """
    return hp.uniform(name, 0.5, 1)


def _lightgbm_colsample_bytree(name):
    """
    Declaration search space 'colsample_bytree' parameter
    """
    return hp.uniform(name, 0.5, 1)


def _lightgbm_reg_alpha(name):
    """
    Declaration search space 'reg_alpha' parameter
    """
    return hp.loguniform(name, np.log(0.0001), np.log(1)) - 0.0001


def _lightgbm_reg_lambda(name):
    """
    Declaration search space 'reg_lambda' parameter
    """
    return hp.loguniform(name, np.log(1), np.log(4))


def _lightgbm_boosting_type(name):
    """
    Declaration search space 'boosting_type' parameter
    """
    return hp.choice(name, ["gbdt", "dart", "goss"])


def _lightgbm_random_state(name: str):
    """
    Declaration search space 'random_state' parameter
    """
    return hp.randint(name, 5)


@validate(params=["boosting_type"],
          validation_test=lambda param: not isinstance(param, str) or param in ["gbdt", "dart", "goss"],
          msg="Invalid parameter '%s' with value '%s'. Value must be one of ['gbdt', 'dart', 'goss'].")
def _lightgbm_hp_space(
        name_func,
        max_depth: typing.Union[int, Apply] = None,
        num_leaves: typing.Union[int, Apply] = None,
        learning_rate: typing.Union[float, Apply] = None,
        n_estimators: typing.Union[int, Apply] = None,
        min_child_weight: typing.Union[float, Apply] = None,
        max_delta_step: float = 0,
        subsample: typing.Union[float, Apply] = None,
        colsample_bytree: typing.Union[float, Apply] = None,
        reg_alpha: typing.Union[float, Apply] = None,
        reg_lambda: typing.Union[float, Apply] = None,
        boosting_type: typing.Union[str, Apply] = None,
        scale_pos_weight: float = 1,
        random_state=None,
        **kwargs):
    """
    Hyper parameter search space for
     lightgbm classifier
     lightgbm regressor
    """
    hp_space = dict(
        # max_depth=_lightgbm_max_depth(name_func("max_depth")) if max_depth is None else max_depth,
        max_depth=-1,
        num_leaves=_lightgbm_num_leaves(name_func("num_leaves")) if num_leaves is None else num_leaves,
        learning_rate=_lightgbm_learning_rate(name_func("learning_rate")) if learning_rate is None else learning_rate,
        n_estimators=_lightgbm_n_estimators(name_func("n_estimators")) if n_estimators is None else n_estimators,
        min_child_weight=_lightgbm_min_child_weight(name_func("min_child_weight"))
        if min_child_weight is None else min_child_weight,
        # min_child_samples=5,
        max_delta_step=max_delta_step,
        subsample=_lightgbm_subsample(name_func("subsample")) if subsample is None else subsample,
        colsample_bytree=_lightgbm_colsample_bytree(name_func("colsample_bytree"))
        if colsample_bytree is None else colsample_bytree,
        reg_alpha=_lightgbm_reg_alpha(name_func("reg_alpha")) if reg_alpha is None else reg_alpha,
        reg_lambda=_lightgbm_reg_lambda(name_func("reg_lambda")) if reg_lambda is None else reg_lambda,
        boosting_type=_lightgbm_boosting_type(name_func("boosting_type")) if boosting_type is None else boosting_type,
        scale_pos_weight=scale_pos_weight,
        seed=_lightgbm_random_state(name_func("random_state")) if random_state is None else random_state,
        **kwargs
    )
    return hp_space


@validate(params=["objective"],
          validation_test=lambda param: not isinstance(param, str) or param in ["binary", "multiclass"],
          msg="Invalid parameter '%s' with value '%s'. Value must be 'binary' or 'multiclass'.")
def lightgbm_classification(name: str, objective: str = None, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a lightgbm.LGBMClassifier model.

    Args:
        name: name | str
        objective: 'binary' or ' multiclass' | str

    See help(hpsklearn.components._lightgbm_hp_space) for info on
    additional available LightGBM arguments.
    """

    def _name(msg):
        return f"{name}.lightgbm_clf_{msg}"

    hp_space = _lightgbm_hp_space(_name, **kwargs)
    hp_space["objective"] = "binary" if objective is None else objective

    return scope.sklearn_LGBMClassifier(**hp_space)


def lightgbm_regression(name: str, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a lightgbm.LightGBMRegressor model.

    Args:
        name: name | str

    See help(hpsklearn.components._lightgbm_hp_space) for info on
    additional available LightGBM arguments.
    """

    def _name(msg):
        return f"{name}.lightgbm_reg_{msg}"

    hp_space = _lightgbm_hp_space(_name, **kwargs)

    return scope.sklearn_LGBMRegressor(objective="regression", **hp_space)

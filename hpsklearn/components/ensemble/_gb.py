from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import ensemble
import numpy as np

import typing


@scope.define
def sklearn_GradientBoostingClassifier(*args, **kwargs):
    return ensemble.GradientBoostingClassifier(*args, **kwargs)


@scope.define
def sklearn_GradientBoostingRegressor(*args, **kwargs):
    return ensemble.GradientBoostingRegressor(*args, **kwargs)


def _gb_clf_loss(name: str):
    """
    Declaration search space 'loss' parameter for _gb classifier
    """
    return hp.choice(name, ["log_loss", "exponential"])


def _gb_reg_loss(name: str):
    """
    Declaration search space 'loss' parameter for _gb regressor
    """
    return hp.choice(name, ["squared_error", "absolute_error", "huber"])


def _gb_reg_alpha(name: str):
    """
    Declaration search space 'alpha' parameter for _gb regressor
    """
    return hp.uniform(name, 0.85, 0.95)


def _gb_learning_rate(name: str):
    """
    Declaration search space 'learning_rate' parameter
    """
    return hp.lognormal(name, np.log(0.01), np.log(10.0))


def _gb_n_estimators(name: str):
    """
    Declaration search space 'n_estimators' parameter
    """
    return scope.int(hp.qloguniform(name, np.log(10.5), np.log(1000.5), 1))


def _gb_criterion(name: str):
    """
    Declaration search space 'criterion' parameter
    """
    return hp.choice(name, ["friedman_mse", "squared_error"])


def _gb_min_samples_split(name: str):
    """
    Declaration search space 'min_samples_split' parameter
    """
    return hp.pchoice(name, [
        (0.95, 2),  # most common choice
        (0.05, 3),  # try minimal increase
    ])


def _gb_min_samples_leaf(name: str):
    """
    Declaration search space 'min_samples_leaf' parameter
    """
    return hp.choice(name, [
        1,  # most common choice.
        scope.int(hp.qloguniform(name + ".gt1", np.log(1.5), np.log(50.5), 1))
    ])


def _gb_min_weight_fraction_leaf(name: str):
    """
    Declaration search space 'min_weight_fraction_leaf' parameter
    """
    return 0.0


def _gb_max_depth(name: str):
    """
    Declaration search space 'max_depth' parameter
    """
    return hp.pchoice(name, [
        (0.1, 2),
        (0.7, 3),  # most common choice.
        (0.1, 4),
        (0.1, 5),
    ])


def _gb_min_impurity_decrease(name: str):
    """
    Declaration search space 'min_impurity_decrease' parameter
    """
    return hp.pchoice(name, [
        (0.85, 0.0),  # most common choice
        (0.05, 0.01),
        (0.05, 0.02),
        (0.05, 0.05),
    ])


def _gb_random_state(name: str):
    """
    Declaration search space 'random_state' parameter
    """
    return hp.randint(name, 5)


def _gb_max_features(name: str):
    """
    Declaration search space 'max_features' parameter
    """
    return hp.pchoice(name, [
        (0.2, "sqrt"),  # most common choice.
        (0.1, "log2"),  # less common choice.
        (0.1, None),  # all features, less common choice.
        (0.6, hp.uniform(name + ".frac", 0., 1.))
    ])


def _gb_max_leaf_nodes(name: str):
    """
    Declaration search space 'max_leaf_nodes' parameter
    """
    return hp.pchoice(name, [
        (0.85, None),  # most common choice
        (0.05, 5),
        (0.05, 10),
        (0.05, 15),
    ])


@validate(params=["max_features"],
          validation_test=lambda param: not isinstance(param, str) or param in ["auto", "sqrt", "log2"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['auto', 'sqrt', 'log2'].")
@validate(params=["n_estimators", "max_depth", "min_samples_split",
                  "min_samples_leaf", "min_weight_fraction_leaf",
                  "min_impurity_decrease", "max_features", "max_leaf_nodes"],
          validation_test=lambda param: not isinstance(param, float) or param > 0,
          msg="Invalid parameter '%s' with value '%s'. Parameter value must be non-negative and greater than 0.")
@validate(params=["ccp_alpha", "learning_rate"],
          validation_test=lambda param: not isinstance(param, float) or 0 <= param <= 1,
          msg="Invalid parameter '%s' with value '%s'. Parameter value must be within [0.0, 1.0].")
def _gb_hp_space(
        name_func,
        learning_rate: typing.Union[float, Apply] = None,
        n_estimators: typing.Union[int, Apply] = None,
        subsample: float = 1.0,
        criterion: typing.Union[str, Apply] = None,
        min_samples_split: typing.Union[float, Apply] = None,
        min_samples_leaf: typing.Union[float, Apply] = None,
        min_weight_fraction_leaf: typing.Union[float, Apply] = None,
        max_depth: typing.Union[int, Apply] = "Undefined",
        min_impurity_decrease: typing.Union[float, Apply] = None,
        init=None,
        random_state=None,
        max_features: typing.Union[str, float, Apply] = "Undefined",
        verbose: int = False,
        max_leaf_nodes: typing.Union[int, Apply] = "Undefined",
        warm_start: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = None,
        tol: float = 1e-4,
        ccp_alpha: float = 0.0,
        **kwargs
):
    """
    Hyper parameter search space for
     gradient boosting classifier
     gradient boosting regressor
    """
    hp_space = dict(
        learning_rate=_gb_learning_rate(name_func("learning_rate")) if learning_rate is None else learning_rate,
        n_estimators=_gb_n_estimators(name_func("n_estimators")) if n_estimators is None else n_estimators,
        subsample=subsample,
        criterion=_gb_criterion(name_func("criterion")) if criterion is None else criterion,
        min_samples_split=_gb_min_samples_split(name_func("min_samples_split"))
        if min_samples_split is None else min_samples_split,
        min_samples_leaf=_gb_min_samples_leaf(name_func("min_samples_leaf"))
        if min_samples_leaf is None else min_samples_leaf,
        min_weight_fraction_leaf=_gb_min_weight_fraction_leaf(name_func("min_weight_fraction_leaf"))
        if min_weight_fraction_leaf is None else min_weight_fraction_leaf,
        max_depth=_gb_max_depth(name_func("max_depth")) if max_depth == "Undefined" else max_depth,
        min_impurity_decrease=_gb_min_impurity_decrease(name_func("min_impurity_decrease"))
        if min_impurity_decrease is None else min_impurity_decrease,
        init=init,
        random_state=_gb_random_state(name_func("random_state")) if random_state is None else random_state,
        max_features=_gb_max_features(name_func("max_features")) if max_features == "Undefined" else max_features,
        verbose=verbose,
        max_leaf_nodes=_gb_max_leaf_nodes(name_func("max_leaf_nodes"))
        if max_leaf_nodes == "Undefined" else max_leaf_nodes,
        warm_start=warm_start,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
        tol=tol,
        ccp_alpha=ccp_alpha,
        **kwargs
    )
    return hp_space


@validate(params=["loss"],
          validation_test=lambda param: not isinstance(param, str) or param in ("log_loss", "exponential"),
          msg="Invalid parameter '%s' with value '%s'. Choose 'log_loss' or 'exponential'.")
def gradient_boosting_classifier(name: str, loss: typing.Union[str, Apply] = None, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.ensemble.GradientBoostingClassifier model.

    Args:
        name: name | str
        loss: choose 'log_loss' or 'exponential' | str

    See help(hpsklearn.components._gb._gb_hp_space) for info on
    additional available GradientBoosting arguments.
    """

    def _name(msg):
        return f"{name}.gbc_{msg}"

    hp_space = _gb_hp_space(_name, **kwargs)
    hp_space["loss"] = _gb_clf_loss(_name("loss")) if loss is None else loss

    return scope.sklearn_GradientBoostingClassifier(**hp_space)


@validate(params=["alpha"],
          validation_test=lambda param: not isinstance(param, float) or 0 <= param <= 1,
          msg="Invalid parameter '%s' with value '%s'. Parameter value must be within [0.0, 1.0]")
@validate(params=["loss"],
          validation_test=lambda param: not isinstance(param, str) or param in ("squared_error", "absolute_error",
                                                                                "huber", "quantile"),
          msg="Invalid parameter '%s' with value '%s'. "
              "Choose 'squared_error', 'absolute_error', 'huber' or 'quantile'.")
def gradient_boosting_regressor(name: str,
                                loss: typing.Union[str, Apply] = None,
                                alpha: typing.Union[float, Apply] = None,
                                **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.ensemble.GradientBoostingRegressor model.

    Args:
        name: name | str
        loss: choose 'squared_error', 'absolute_error', 'huber' or 'quantile' | str
        alpha: alpha parameter for huber and quantile loss | float

    See help(hpsklearn.components._gb._gb_hp_space) for info on
    additional available GradientBoosting arguments.
    """
    if isinstance(loss, str) and isinstance(alpha, float) and loss not in ("quantile", "huber"):
        raise ValueError("The 'alpha' parameter can only be specified for 'loss': 'huber' or 'quantile'.")

    if isinstance(alpha, float) and not isinstance(loss, str):
        raise ValueError("For custom 'alpha' parameter, the 'loss' parameter must be specified.")

    def _name(msg):
        return f"{name}.gbr_{msg}"

    hp_space = _gb_hp_space(_name, **kwargs)
    hp_space["loss"] = _gb_reg_loss(_name("loss")) if loss is None else loss
    hp_space["alpha"] = _gb_reg_alpha(_name("alpha")) if alpha is None else alpha

    return scope.sklearn_GradientBoostingRegressor(**hp_space)

from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import ensemble
import numpy as np
import numpy.typing as npt

import typing


@scope.define
def sklearn_HistGradientBoostingClassifier(*args, **kwargs):
    return ensemble.HistGradientBoostingClassifier(*args, **kwargs)


@scope.define
def sklearn_HistGradientBoostingRegressor(*args, **kwargs):
    return ensemble.HistGradientBoostingRegressor(*args, **kwargs)


def _hist_gradient_boosting_reg_loss(name: str):
    """
    Declaration of search space 'criterion' parameter for
     hist gradient boosting regressor

    Parameter 'poisson' is also available. Not implemented since
     'poisson' is only available for non-zero, non-negative y data

    Parameter 'gamma' is also available. Not implemented since
     'gamma' is only available for non-negative y data
    """
    return hp.choice(name, ["squared_error", "absolute_error"])


def _hist_gradient_boosting_learning_rate(name: str):
    """
    Declaration search space 'learning_rate' parameter
    """
    return hp.lognormal(name, np.log(0.01), np.log(10.0))


def _hist_gradient_boosting_max_leaf_nodes(name: str):
    """
    Declaration search space 'max_leaf_nodes' parameter
    """
    return scope.int(hp.qnormal(name, 31, 5, 1))


def _hist_gradient_boosting_max_depth(name: str):
    """
    Declaration search space 'max_depth' parameter
    """
    return hp.pchoice(name, [
        (0.15, 2),
        (0.7, 3),  # most common choice.
        (0.15, 4),
    ])


def _hist_gradient_boosting_min_samples_leaf(name: str):
    """
    Declaration search space 'min_samples_leaf' parameter
    """
    return scope.int(hp.qnormal(name, 20, 2, 1))


def _hist_gradient_boosting_random_state(name: str):
    """
    Declaration search space 'random_state' parameter
    """
    return hp.randint(name, 5)


def _hist_gradient_boosting_max_features(name: str):
    """
    Declaration search space 'max_features' parameter
    """
    return hp.uniform(name + ".frac", 0.5, 1.)


@validate(params=["max_bins"],
          validation_test=lambda param: not isinstance(param, int) or 0 < param <= 255,
          msg="Invalid parameter '%s' with value '%s'. "
              "Parameter value must be Parameter value must be within (0, 255].")
@validate(params=["max_leaf_nodes"],
          validation_test=lambda param: not isinstance(param, int) or param > 1,
          msg="Invalid parameter '%s' with value '%s'. "
              "Parameter value must be strictly higher than 1.")
def _hist_gradient_boosting_hp_space(
        name_func,
        learning_rate: typing.Union[float, Apply] = None,
        max_iter: typing.Union[int, Apply] = 100,
        max_leaf_nodes: typing.Union[int, Apply] = "Undefined",
        max_depth: typing.Union[int, Apply] = "Undefined",
        min_samples_leaf: typing.Union[int, Apply] = None,
        l2_regularization: float = 0,
        max_features: typing.Union[float, Apply] = None,
        max_bins: int = 255,
        categorical_features: npt.ArrayLike = None,
        monotonic_cst: npt.ArrayLike = None,
        warm_start: bool = False,
        early_stopping: typing.Union[str, bool] = "auto",
        scoring: typing.Union[str, callable] = "loss",
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
        tol: float = 1e-7,
        verbose: int = False,
        random_state=None,
        **kwargs
):
    """
    Hyper parameter search space for
     hist gradient boosting classifier
     hist gradient boosting regressor
    """
    if not early_stopping and (isinstance(scoring, str | callable) or
                               isinstance(validation_fraction, float) or
                               isinstance(n_iter_no_change, int)):
        raise ValueError("Invalid declaration of parameters."
                         "Parameters 'scoring', 'validation_fraction' and 'n_iter_no_change' "
                         "can only be specified in addition to 'early_stopping'.")
    hp_space = dict(
        learning_rate=_hist_gradient_boosting_learning_rate(name_func("learning_rate"))
        if learning_rate is None else learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=_hist_gradient_boosting_max_leaf_nodes(name_func("max_leaf_nodes"))
        if max_leaf_nodes == "Undefined" else max_leaf_nodes,
        max_depth=_hist_gradient_boosting_max_depth(name_func("max_depth"))
        if max_depth == "Undefined" else max_depth,
        min_samples_leaf=_hist_gradient_boosting_min_samples_leaf(name_func("min_samples_leaf"))
        if min_samples_leaf is None else min_samples_leaf,
        l2_regularization=l2_regularization,
        max_features=_hist_gradient_boosting_max_features(name_func("max_features"))
        if max_features is None else max_features,
        max_bins=max_bins,
        categorical_features=categorical_features,
        monotonic_cst=monotonic_cst,
        warm_start=warm_start,
        early_stopping=early_stopping,
        scoring=scoring,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
        tol=tol,
        verbose=verbose,
        random_state=_hist_gradient_boosting_random_state(name_func("random_state"))
        if random_state is None else random_state,
        **kwargs
    )
    return hp_space


@validate(params=["loss"],
          validation_test=lambda param: not isinstance(param, str) or param in ("auto", "binary_crossentropy",
                                                                                "categorical_crossentropy"),
          msg="Invalid parameter '%s' with value '%s'. "
              "Choose 'auto', 'binary_crossentropy', 'categorical_crossentropy'")
def hist_gradient_boosting_classifier(name: str, loss: typing.Union[str, Apply] = "log_loss", **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.ensemble.HistGradientBoostingClassifier model.

    Args:
        name: name | str
        loss: 'log_loss' | str

    See help(hpsklearn.components._hist_gradient_boosting._hist_gradient_boosting_regressor) for info on
    additional available HistGradientBoosting arguments.
    """

    def _name(msg):
        return f"{name}.gbc_{msg}"

    hp_space = _hist_gradient_boosting_hp_space(_name, **kwargs)
    hp_space["loss"] = loss

    return scope.sklearn_HistGradientBoostingClassifier(**hp_space)


@validate(params=["loss"],
          validation_test=lambda param: not isinstance(param, str) or param in ("squared_error", "absolute_error",
                                                                                "poisson", "quantile", "gamma"),
          msg="Invalid parameter '%s' with value '%s'. "
              "Choose 'squared_error', 'absolute_error', 'poisson'")
def hist_gradient_boosting_regressor(name: str, loss: typing.Union[str, Apply] = None, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.ensemble.HistGradientBoostingRegressor model.

    Args:
        name: name | str
        loss: choose 'squared_error', 'absolute_error' or 'poisson' | str

    See help(hpsklearn.components._hist_gradient_boosting._hist_gradient_boosting_regressor) for info on
    additional available HistGradientBoosting arguments.
    """

    def _name(msg):
        return f"{name}.gbc_{msg}"

    hp_space = _hist_gradient_boosting_hp_space(_name, **kwargs)
    hp_space["loss"] = _hist_gradient_boosting_reg_loss(_name("loss")) if loss is None else loss

    return scope.sklearn_HistGradientBoostingRegressor(**hp_space)

from hpsklearn.components._base import validate

from hyperopt.pyll import scope
from hyperopt import hp

from sklearn import linear_model
import numpy as np
import typing


@scope.define
def sklearn_PassiveAggressiveClassifier(*args, **kwargs):
    return linear_model.PassiveAggressiveClassifier(*args, **kwargs)


@scope.define
def sklearn_PassiveAggressiveRegressor(*args, **kwargs):
    return linear_model.PassiveAggressiveRegressor(*args, **kwargs)


def _passive_aggressive_C(name: str):
    """
    Declaration search space 'C' parameter
    """
    return hp.uniform(name, 0.1, 2)


def _passive_aggressive_max_iter(name: str):
    """
    Declaration search space 'max_iter' parameter
    """
    return hp.qloguniform(name, low=np.log(750), high=np.log(1250), q=1.0)


def _passive_aggressive_tol(name: str):
    """
    Declaration search space 'tol' parameter
    """
    return hp.loguniform(name, np.log(1e-5), np.log(1e-2))


def _passive_aggressive_n_iter_no_change(name: str):
    """
    Declaration search space 'n_iter_no_change' parameter
    """
    return hp.pchoice(name, [
        (0.25, 4),
        (0.50, 5),
        (0.25, 6)
    ])


def _passive_aggressive_random_state(name: str):
    """
    Declaration search space 'random_state' parameter
    """
    return hp.randint(name, 5)


def _passive_aggressive_hp_space(
        name_func,
        C: float = None,
        fit_intercept: bool = True,
        max_iter: int = None,
        tol: float = None,
        early_stopping: bool = False,
        validation_fraction: float = None,
        n_iter_no_change: int = None,
        shuffle: bool = True,
        verbose: int = 0,
        random_state=None,
        warm_start: bool = False,
        average: typing.Union[bool, int] = False
):
    """
    Hyper parameter search space for
     passive aggressive classifier
     passive aggressive regressor
    """
    hp_space = dict(
        C=_passive_aggressive_C(name_func("C")) if C is None else C,
        fit_intercept=fit_intercept,
        max_iter=max_iter or _passive_aggressive_max_iter(name_func("max_iter")),
        tol=_passive_aggressive_tol(name_func("tol")) if tol is None else tol,
        early_stopping=early_stopping,
        validation_fraction=0.1 if validation_fraction is None else validation_fraction,
        n_iter_no_change=n_iter_no_change or _passive_aggressive_n_iter_no_change(name_func("n_iter_no_change")),
        shuffle=shuffle,
        verbose=verbose,
        random_state=_passive_aggressive_random_state(name_func("random_state"))
        if random_state is None else random_state,
        warm_start=warm_start,
        average=average
    )
    return hp_space


def passive_aggressive_classifier(name: str,
                                  loss: str = None,
                                  n_jobs: int = 1,
                                  class_weight: typing.Union[dict, str] = None,
                                  **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.PassiveAggressiveClassifier model.

    Args:
        name: name | str
        loss: loss function to use | str
        n_jobs: number of CPUs to use | int
        class_weight: class_weight fit parameters | dict or str

    See help(hpsklearn.components.linear_model._passive_aggressive._passive_aggressive_hp_space)
    for info on additional available passive aggressive arguments.
    """
    def _name(msg):
        return f"{name}.passive_aggressive_classifier{msg}"

    hp_space = _passive_aggressive_hp_space(_name, **kwargs)
    hp_space["loss"] = loss or hp.choice(_name("loss"), ["hinge", "squared_hinge"])
    hp_space["n_jobs"] = n_jobs
    hp_space["class_weight"] = class_weight

    return scope.sklearn_PassiveAggressiveClassifier(**hp_space)


def passive_aggressive_regressor(name: str,
                                 loss: str = None,
                                 epsilon: float = None,
                                 **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.PassiveAggressiveRegressor model.

    Args:
        name: name | str
        loss: loss function to use | str
        epsilon: threshold prediction and correct | float

    See help(hpsklearn.components.linear_model._passive_aggressive._passive_aggressive_hp_space)
    for info on additional available passive aggressive arguments.
    """
    def _name(msg):
        return f"{name}.passive_aggressive_classifier{msg}"

    hp_space = _passive_aggressive_hp_space(_name, **kwargs)
    hp_space["loss"] = loss or hp.choice(_name("loss"), ["epsilon_insensitive", "squared_epsilon_insensitive"])
    hp_space["epsilon"] = hp.uniform(_name("epsilon"), 0.05, 0.2) if epsilon is None else epsilon

    return scope.sklearn_PassiveAggressiveRegressor(**hp_space)

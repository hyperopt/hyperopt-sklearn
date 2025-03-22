from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import linear_model
import numpy as np
import typing


@scope.define
def sklearn_LogisticRegression(*args, **kwargs):
    return linear_model.LogisticRegression(*args, **kwargs)


@scope.define
def sklearn_LogisticRegressionCV(*args, **kwargs):
    return linear_model.LogisticRegressionCV(*args, **kwargs)


def _logistic_penalty_solver(name: str):
    """
    Declaration search space 'penalty' and 'solver' parameters
    """
    solver = np.random.choice(["newton-cg", "lbfgs", "liblinear", "sag", "saga"])
    if solver in ["newton-cg", "lbfgs", "sag"]:
        penalty = "l2"
    elif solver == "liblinear":
        penalty = hp.choice(name, ["l1", "l2"])
    else:
        penalty = hp.choice(name, ["elasticnet", "l1", "l2"])
    return penalty, solver


def _logistic_tol(name: str):
    """
    Declaration search space 'tol' parameter
    """
    return hp.loguniform(name, np.log(1e-5), np.log(1e-2))


def _logistic_max_iter(name: str):
    """
    Declaration search space 'max_iter' parameter
    """
    return scope.int(hp.uniform(name, 500, 1000))


def _logistic_C(name: str):
    """
    Declaration search space 'C' parameter
    """
    return hp.uniform(name, 0.1, 2)


def _logistic_Cs(name: str):
    """
    Declaration search space 'Cs' parameter
    """
    return scope.int(hp.normal(name, mu=10, sigma=2))


def _logistic_cv(name: str):
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


def _logistic_random_state(name: str):
    """
    Declaration search space 'random_state' parameter
    """
    return hp.randint(name, 5)


@validate(params=["penalty"],
          validation_test=lambda param: not isinstance(param, str) or param in ["l1", "l2", "elasticnet"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['l1', 'l2', 'elasticnet'].")
@validate(params=["solver"],
          validation_test=lambda param: not isinstance(param, str) or param in ["newton-cg", "lbfgs", "liblinear",
                                                                                "sag", "saga"],
          msg="Invalid parameter '%s' with value '%s'. "
              "Value must be in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'].")
def _logistic_hp_space(
        name_func,
        fit_intercept: bool = True,
        dual: bool = False,
        penalty: typing.Union[str, Apply] = None,
        solver: typing.Union[str, Apply] = None,
        tol: typing.Union[float, Apply] = None,
        max_iter: typing.Union[int, Apply] = None,
        class_weight: typing.Union[dict, str] = None,
        n_jobs: int = 1,
        verbose: int = 0,
        intercept_scaling: float = 1,
        random_state=None,
        **kwargs
):
    if dual is True and not (penalty == "l2" and solver == "liblinear"):
        raise ValueError("Dual formulation (implied by parameter 'dual' = 'True') is only implemented for "
                         "'l2' penalty with 'liblinear' solver.")

    if penalty is None or solver is None:
        penalty, solver = _logistic_penalty_solver(name_func("penalty_solver"))

    hp_space = dict(
        fit_intercept=fit_intercept,
        dual=dual,
        penalty=penalty,
        solver=solver,
        tol=_logistic_tol(name_func("tol")) if tol is None else tol,
        max_iter=_logistic_max_iter(name_func("max_iter")) if max_iter is None else max_iter,
        class_weight=class_weight,
        n_jobs=n_jobs,
        verbose=verbose,
        intercept_scaling=intercept_scaling,
        random_state=_logistic_random_state(name_func("random_state"))
        if random_state is None else random_state,
        **kwargs,
    )
    return hp_space


@validate(params=["l1_ratio"],
          validation_test=lambda param: not isinstance(param, int) or 0 > param > 1,
          msg="Invalid parameter '%s' with value '%s'. Value must be between 0 and 1.")
def logistic_regression(name: str,
                        C: typing.Union[float, Apply] = None,
                        warm_start: bool = False,
                        l1_ratio: float = None,
                        **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.LogisticRegression model.

    Args:
        name: name | str
        C: inverse of regularization strength | float
        warm_start: whether to reuse solution of previous call | bool
        l1_ratio: elasticnet mixing parameter

    See help(hpsklearn.components.linear_model._logistic._logistic_hp_space)
    for info on additional available logistic arguments.
    """

    def _name(msg):
        return f"{name}.logistic_regression_{msg}"

    hp_space = _logistic_hp_space(_name, **kwargs)
    hp_space["C"] = _logistic_C(_name("C")) if C is None else C
    hp_space["warm_start"] = warm_start
    hp_space["l1_ratio"] = l1_ratio

    return scope.sklearn_LogisticRegression(**hp_space)


def logistic_regression_cv(name: str,
                           Cs: typing.Union[int, typing.List[float], Apply] = None,
                           cv: typing.Union[int, typing.Generator, Apply] = None,
                           scoring: typing.Union[str, callable] = None,
                           refit: bool = True,
                           l1_ratios: typing.List[float] = None,
                           **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.LogisticRegressionCV model.

    Args:
        name: name | str
        Cs: list of inverse of regularization strength | int, list of floats
        cv: cross-validation generator | int, generator
        scoring: scorer | str, callable
        refit: average scores across folds | bool
        l1_ratios: list of elastic net mixing parameter | list of floats

    See help(hpsklearn.components.linear_model._logistic._logistic_hp_space)
    for info on additional available logistic arguments.
    """

    def _name(msg):
        return f"{name}.logistic_regression_cv_{msg}"

    hp_space = _logistic_hp_space(_name, **kwargs)
    hp_space["Cs"] = _logistic_Cs(_name("Cs")) if Cs is None else Cs
    hp_space["cv"] = _logistic_cv(_name("cv")) if cv is None else cv
    hp_space["scoring"] = scoring
    hp_space["refit"] = refit
    hp_space["l1_ratios"] = l1_ratios

    return scope.sklearn_LogisticRegressionCV(**hp_space)

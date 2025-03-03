from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import ensemble
import numpy as np

import typing


@scope.define
def sklearn_IsolationForest(*args, **kwargs):
    return ensemble.IsolationForest(*args, **kwargs)


def _iforest_n_estimators(name: str):
    """
    Declaration search space 'n_estimators' parameter
    """
    return scope.int(hp.qloguniform(name, np.log(9.5), np.log(3000.5), 1))


def _iforest_max_features(name: str):
    """
    Declaration search space 'max_features' parameter
    """
    return hp.pchoice(name, [
        (0.05, 0.8),
        (0.15, 0.9),
        (0.8, 1.0),  # most common choice
    ])


def _iforest_bootstrap(name: str):
    """
    Declaration search space 'bootstrap' parameter
    """
    return hp.choice(name, [True, False])


def _iforest_random_state(name: str):
    """
    Declaration search space 'random_state' parameter
    """
    return hp.randint(name, 5)


@validate(params=["contamination"],
          validation_test=lambda param: not isinstance(param, float) or 0 < param <= .5,
          msg="Invalid parameter '%s' with value '%s'. Parameter value should be in the range (0, 0.5].")
def _iforest_hp_space(
        name_func,
        n_estimators: typing.Union[int, Apply] = None,
        max_samples: typing.Union[str, float, Apply] = "auto",
        contamination: typing.Union[str, float, Apply] = "auto",
        max_features: typing.Union[float, Apply] = None,
        bootstrap: typing.Union[bool, Apply] = None,
        n_jobs: int = 1,
        random_state=None,
        verbose: int = False,
        warm_start: bool = False,
        **kwargs
):
    """
    Hyper parameter search space for
     isolation forest algorithm
    """
    hp_space = dict(
        n_estimators=_iforest_n_estimators(name_func("n_estimators")) if n_estimators is None else n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        max_features=_iforest_max_features(name_func("max_features")) if max_features is None else max_features,
        bootstrap=_iforest_bootstrap(name_func("bootstrap")) if bootstrap is None else bootstrap,
        n_jobs=n_jobs,
        random_state=_iforest_random_state(name_func("random_state")) if random_state is None else random_state,
        verbose=verbose,
        warm_start=warm_start,
        **kwargs
    )
    return hp_space


def isolation_forest(name: str, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.ensemble.IsolationForest model.

    Args:
        name: name | str

    See help(hpsklearn.components.ensemble._iforest._iforest_hp_space)
    for info on additional available bagging arguments.
    """

    def _name(msg):
        return f"{name}.if_{msg}"

    hp_space = _iforest_hp_space(_name, **kwargs)

    return scope.sklearn_IsolationForest(**hp_space)

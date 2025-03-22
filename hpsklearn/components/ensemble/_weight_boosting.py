from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import ensemble
import numpy as np
import typing


@scope.define
def sklearn_AdaBoostClassifier(*args, **kwargs):
    return ensemble.AdaBoostClassifier(*args, **kwargs)


@scope.define
def sklearn_AdaBoostRegressor(*args, **kwargs):
    return ensemble.AdaBoostRegressor(*args, **kwargs)


def _weight_boosting_n_estimators(name: str):
    """
    Declaration search space 'n_estimators' parameter
    """
    return scope.int(hp.qloguniform(name, np.log(10.5), np.log(1000.5), 1))


def _weight_boosting_learning_rate(name: str):
    """
    Declaration search space 'learning_rate' parameter
    """
    return hp.lognormal(name, np.log(0.01), np.log(10.0))


def _weight_boosting_loss(name: str):
    """
    Declaration search space 'loss' parameter
    """
    return hp.choice(name, ["linear", "square", "exponential"])


def _weight_boosting_random_state(name: str):
    """
    Declaration search space 'random_state' parameter
    """
    return hp.randint(name, 5)


@validate(params=["n_estimators", "learning_rate"],
          validation_test=lambda param: not isinstance(param, float) or param > 0,
          msg="Invalid parameter '%s' with value '%s'. Parameter value must be non-negative and greater than 0.")
def _weight_boosting_hp_space(
    name_func,
    estimator=None,
    n_estimators: typing.Union[int, Apply] = None,
    learning_rate: typing.Union[float, Apply] = None,
    random_state=None,
    **kwargs
):
    """
    Hyper parameter search space for
     AdaBoost classifier
     AdaBoost regressor
    """
    hp_space = dict(
        estimator=estimator,
        n_estimators=_weight_boosting_n_estimators(name_func("n_estimators")) if n_estimators is None else n_estimators,
        learning_rate=_weight_boosting_learning_rate(name_func("learning_rate"))
        if learning_rate is None else learning_rate,
        random_state=_weight_boosting_random_state(name_func("random_state")) if random_state is None else random_state,
        **kwargs
    )
    return hp_space


def ada_boost_classifier(name: str, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.ensemble.AdaBoostClassifier model.

    Args:
        name: name | str

    See help(hpsklearn.components.ensemble._weight_boosting._weight_boosting_hp_space)
    for info on additional available AdaBoost arguments.
    """

    def _name(msg):
        return f"{name}.ada_boost_{msg}"

    hp_space = _weight_boosting_hp_space(_name, **kwargs)

    return scope.sklearn_AdaBoostClassifier(**hp_space)


def ada_boost_regressor(name: str, loss: typing.Union[str, Apply] = None, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.ensemble.AdaBoostClassifier model.

    Args:
        name: name | str
        loss: choose 'linear', 'square' or 'exponential' | str

    See help(hpsklearn.components.ensemble._weight_boosting._weight_boosting_hp_space)
    for info on additional available AdaBoost arguments.
    """

    def _name(msg):
        return f"{name}.ada_boost_{msg}"

    hp_space = _weight_boosting_hp_space(_name, **kwargs)
    hp_space["loss"] = _weight_boosting_loss(_name("loss")) if loss is None else loss

    return scope.sklearn_AdaBoostRegressor(**hp_space)

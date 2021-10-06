from ._base import validate

from hyperopt.pyll import scope
from hyperopt import hp

from sklearn import ensemble


@scope.define
def sklearn_BaggingClassifier(*args, **kwargs):
    return ensemble.BaggingClassifier(*args, **kwargs)


@scope.define
def sklearn_BaggingRegressor(*args, **kwargs):
    return ensemble.BaggingRegressor(*args, **kwargs)


def _bagging_n_estimators(name):
    """
    Declaration search space 'n_estimators' parameter
    """
    return hp.pchoice(name, [
        # create custom distribution
        (0.0625, 8),
        (0.175, 9),
        (0.525, 10),  # most common choice
        (0.175, 11),
        (0.0625, 12),
    ])


def _bagging_max_samples(name):
    """
    Declaration search space 'max_samples' parameter
    """
    return hp.pchoice(name, [
        (0.05, 0.8),
        (0.15, 0.9),
        (0.8, 1.0),  # most common choice
    ])


def _bagging_max_features(name):
    """
    Declaration search space 'max_features' parameter
    """
    return hp.pchoice(name, [
        (0.05, 0.8),
        (0.15, 0.9),
        (0.8, 1.0),  # most common choice
    ])


def _bagging_bootstrap(name):
    """
    Declaration search space 'bootstrap' parameter
    """
    return hp.choice(name, [True, False])


def _bagging_bootstrap_features(name):
    """
    Declaration search space 'bootstrap_features' parameter
    """
    return hp.choice(name, [True, False])


def _bagging_random_state(name):
    """
    Declaration search space 'random_state' parameter
    """
    return hp.randint(name, 5)


@validate(params=["max_samples", "max_features"],
          validation_test=lambda param: isinstance(param, float) and not param > 0,
          msg="Invalid parameter '%s' with value '%s'. Parameter value must be non-negative and greater than 0.")
@validate(params=["n_estimators"],
          validation_test=lambda param: isinstance(param, int) and not param > 1,
          msg="Invalid parameter '%s' with value '%s'. Parameter value must exceed 1.")
def _bagging_hp_space(
        name_func,
        base_estimator=None,
        n_estimators: int = None,
        max_samples: float = None,
        max_features: float = None,
        bootstrap: bool = None,
        bootstrap_features: bool = None,
        oob_score: bool = False,
        warm_start: bool = False,
        n_jobs: int = 1,
        random_state=None,
        verbose: int = False,
):
    """
    Hyper parameter search space for
     bagging classifier
     bagging regressor
    """
    hp_space = dict(
        base_estimator=base_estimator,
        n_estimators=(n_estimators or _bagging_n_estimators(name_func("n_estimators"))),
        max_samples=(max_samples or _bagging_max_samples(name_func("max_samples"))),
        max_features=(max_features or _bagging_max_features(name_func("max_features"))),
        bootstrap=(bootstrap or _bagging_bootstrap(name_func("bootstrap"))),
        bootstrap_features=(bootstrap_features or _bagging_bootstrap_features(name_func("bootstrap_features"))),
        oob_score=oob_score,
        warm_start=warm_start,
        n_jobs=n_jobs,
        random_state=_bagging_random_state(name_func("random_state")) if random_state is None else random_state,
        verbose=verbose,
    )
    return hp_space


def bagging_classifier(name: str, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.ensemble.BaggingClassifier model.

    Args:
        name: name | str

    See help(hpsklearn.components.ensemble._bagging._bagging_hp_space)
    for info on additional available bagging arguments.
    """
    def _name(msg):
        return f"{name}.bc_{msg}"

    hp_space = _bagging_hp_space(_name, **kwargs)

    return scope.sklearn_BaggingClassifier(**hp_space)


def bagging_regressor(name: str, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.ensemble.BaggingRegressor model.

    Args:
        name: name | str

    See help(hpsklearn.components.ensemble._bagging._bagging_hp_space)
    for info on additional available bagging arguments.
    """
    def _name(msg):
        return f"{name}.br_{msg}"

    hp_space = _bagging_hp_space(_name, **kwargs)

    return scope.sklearn_BaggingRegressor(**hp_space)

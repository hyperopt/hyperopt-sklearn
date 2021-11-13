from hpsklearn.components._base import validate

from hyperopt.pyll import scope
from hyperopt import hp

from sklearn import dummy
import numpy.typing as npt
import typing


@scope.define
def sklearn_DummyClassifier(*args, **kwargs):
    return dummy.DummyClassifier(*args, **kwargs)


@scope.define
def sklearn_DummyRegressor(*args, **kwargs):
    return dummy.DummyRegressor(*args, **kwargs)


@validate(params=["strategy"],
          validation_test=lambda param: isinstance(param, str) and param in ["stratified", "most_frequent", "prior",
                                                                             "uniform", "constant"],
          msg="Invalid parameter '%s' with value '%s'. "
              "Value must be in ['stratified', 'most_frequent', 'prior', 'uniform', 'constant'].")
def dummy_classifier(name: str,
                     strategy: str = None,
                     random_state=None,
                     constant: typing.Union[int, str, npt.ArrayLike] = None):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.dummy.DummyClassifier model.

    Args:
        name: name | str
        strategy: strategy to generate predictions | str
        random_state: randomness control predictions
        constant: constant for 'constant' strategy | int, str, npt.ArrayLike
    """
    def _name(msg):
        return f"{name}.dummy_classifier_{msg}"

    hp_space = dict(
        strategy=strategy or hp.choice(_name("strategy"), ["stratified", "most_frequent", "prior", "uniform"]),
        random_state=hp.randint(_name("random_state"), 5) if random_state is None else random_state,
        constant=constant
    )
    return scope.sklearn_DummyClassifier(**hp_space)


@validate(params=["strategy"],
          validation_test=lambda param: isinstance(param, str) and param in ["mean", "median", "quantile", "constant"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['mean', 'median', 'quantile', 'constant'].")
@validate(params=["quantile"],
          validation_test=lambda param: isinstance(param, float) and 0 >= param >= 1,
          msg="Invalid parameter '%s' with value '%s'. Value must be between [0.0, 1.0].")
def dummy_regressor(name: str,
                    strategy: str = None,
                    constant: typing.Union[int, str, npt.ArrayLike] = None,
                    quantile: float = None):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.dummy.DummyRegressor model.

    Args:
        name: name | str
        strategy: strategy to generate predictions | str
        constant: constant for 'constant' strategy | int, str, npt.ArrayLike
        quantile: quantile for 'quantile' strategy | float
    """
    def _name(msg):
        return f"{name}.dummy_regressor_{msg}"

    hp_space = dict(
        strategy=strategy or hp.choice(_name("strategy"), ["mean", "median"]),
        constant=constant,
        quantile=quantile
    )
    return scope.sklearn_DummyRegressor(**hp_space)

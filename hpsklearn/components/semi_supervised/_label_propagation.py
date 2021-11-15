from hpsklearn.components._base import validate

from hyperopt.pyll import scope
from hyperopt import hp

from sklearn import semi_supervised
import numpy as np
import typing


@scope.define
def sklearn_LabelPropagation(*args, **kwargs):
    return semi_supervised.LabelPropagation(*args, **kwargs)


@scope.define
def sklearn_LabelSpreading(*args, **kwargs):
    return semi_supervised.LabelSpreading(*args, **kwargs)


def _label_propagation_kernel(name: str):
    """
    Declaration search space 'kernel' parameter
    """
    return hp.choice(name, ["knn", "rbf"])


def _label_propagation_gamma(name: str):
    """
    Declaration search space 'gamma' parameter
    """
    return hp.uniform(name, 10, 30)


def _label_propagation_n_neighbors(name: str):
    """
    Declaration search space 'n_neighbors' parameter
    """
    return scope.int(hp.uniform(name, 3, 11))


def _label_propagation_n_tol(name: str):
    """
    Declaration search space 'tol' parameter
    """
    return hp.loguniform(name, np.log(1e-5), np.log(1e-2))


@validate(params=["gamma", "n_neighbors", "tol"],
          validation_test=lambda param: isinstance(param, float) and param > 0,
          msg="Invalid parameter '%s' with value '%s'. Parameter value must be non-negative and greater than 0.")
def _label_propagation_hp_space(
        name_func,
        kernel: typing.Union[str, callable] = None,
        gamma: float = None,
        n_neighbors: int = None,
        tol: float = None,
        n_jobs: int = 1
):
    """
    Hyper parameter search space for
     label propagation
     label spreading
    """
    hp_space = dict(
        kernel=kernel or _label_propagation_kernel(name_func("kernel")),
        gamma=gamma or _label_propagation_gamma(name_func("gamma")),
        n_neighbors=n_neighbors or _label_propagation_n_neighbors(name_func("n_neighbors")),
        tol=tol or _label_propagation_n_tol(name_func("tol")),
        n_jobs=n_jobs
    )
    return hp_space


def label_propagation(name: str, max_iter: int = None, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.semi_supervised.LabelPropagation model.

    Args:
        name: name | str
        max_iter: maximum iterations | int

    See help(hpsklearn.components.semi_supervised._label_propagation._label_propagation_hp_space)
    for info on additional available label propagation arguments.
    """
    def _name(msg):
        return f"{name}.label_propagation_{msg}"

    hp_space = _label_propagation_hp_space(_name, **kwargs)
    hp_space["max_iter"] = max_iter or scope.int(hp.uniform(_name("max_iter"), 750, 1250))

    return scope.sklearn_LabelPropagation(**hp_space)


def label_spreading(name: str, alpha: float = None, max_iter: int = None, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.semi_supervised.LabelPropagation model.

    Args:
        name: name | str
        alpha: clamping factor | float
        max_iter: maximum iterations | int

    See help(hpsklearn.components.semi_supervised._label_propagation._label_propagation_hp_space)
    for info on additional available label propagation arguments.
    """
    def _name(msg):
        return f"{name}.label_spreading_{msg}"

    hp_space = _label_propagation_hp_space(_name, **kwargs)
    hp_space["alpha"] = alpha or hp.uniform(_name("alpha"), 0.1, 0.9)
    hp_space["max_iter"] = max_iter or scope.int(hp.uniform(_name("max_iter"), 10, 50))

    return scope.sklearn_LabelSpreading(**hp_space)

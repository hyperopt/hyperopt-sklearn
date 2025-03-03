from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
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
          validation_test=lambda param: not isinstance(param, float) or param > 0,
          msg="Invalid parameter '%s' with value '%s'. Parameter value must be non-negative and greater than 0.")
def _label_propagation_hp_space(
        name_func,
        kernel: typing.Union[str, callable, Apply] = None,
        gamma: typing.Union[float, Apply] = None,
        n_neighbors: typing.Union[int, Apply] = None,
        tol: typing.Union[float, Apply] = None,
        n_jobs: int = 1,
        **kwargs
):
    """
    Hyper parameter search space for
     label propagation
     label spreading
    """
    hp_space = dict(
        kernel=_label_propagation_kernel(name_func("kernel")) if kernel is None else kernel,
        gamma=_label_propagation_gamma(name_func("gamma")) if gamma is None else gamma,
        n_neighbors=_label_propagation_n_neighbors(name_func("n_neighbors")) if n_neighbors is None else n_neighbors,
        tol=_label_propagation_n_tol(name_func("tol")) if tol is None else tol,
        n_jobs=n_jobs,
        **kwargs
    )
    return hp_space


def label_propagation(name: str, max_iter: typing.Union[int, Apply] = None, **kwargs):
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
    hp_space["max_iter"] = scope.int(hp.uniform(_name("max_iter"), 750, 1250)) if max_iter is None else max_iter

    return scope.sklearn_LabelPropagation(**hp_space)


def label_spreading(name: str,
                    alpha: typing.Union[float, Apply] = None,
                    max_iter: typing.Union[int, Apply] = None,
                    **kwargs):
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
    hp_space["alpha"] = hp.uniform(_name("alpha"), 0.1, 0.9) if alpha is None else alpha
    hp_space["max_iter"] = scope.int(hp.uniform(_name("max_iter"), 10, 50)) if max_iter is None else max_iter

    return scope.sklearn_LabelSpreading(**hp_space)

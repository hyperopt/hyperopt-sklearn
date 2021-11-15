from hpsklearn.components._base import validate

from hyperopt.pyll import scope
from hyperopt import hp

from sklearn import cross_decomposition
import numpy as np


@scope.define
def sklearn_CCA(*args, **kwargs):
    return cross_decomposition.CCA(*args, **kwargs)


@scope.define
def sklearn_PLSCanonical(*args, **kwargs):
    return cross_decomposition.PLSCanonical(*args, **kwargs)


@scope.define
def sklearn_PLSRegression(*args, **kwargs):
    return cross_decomposition.PLSRegression(*args, **kwargs)


def _pls_n_components(name: str):
    """
    Declaration search space 'n_components' parameter
    """
    return hp.choice(name, [1, 2])


def _pls_max_iter(name: str):
    """
    Declaration search space 'max_iter' parameter
    """
    scope.int(hp.qloguniform(name, np.log(350), np.log(650), 1))


def _pls_tol(name: str):
    """
    Declaration search space 'tol' parameter
    """
    return hp.loguniform(name, np.log(1e-7), np.log(1e-5))


def _pls_hp_space(
        name_func,
        n_components: int = None,
        scale: bool = True,
        max_iter: int = None,
        tol: float = None,
        copy: bool = True
):
    """
    Hyper parameter search space for
     cca
     pls canonical
     pls regression
    """
    hp_space = dict(
        n_components=_pls_n_components(name_func("n_components")) if n_components is None else n_components,
        scale=scale,
        max_iter=_pls_max_iter(name_func("max_iter")) if max_iter is None else max_iter,
        tol=_pls_tol(name_func("tol")) if tol is None else tol,
        copy=copy
    )
    return hp_space


def cca(name: str, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.cross_decomposition.CCA model.

    Args:
        name: name | str

    See help(hpsklearn.components.cross_decomposition._pls._pls_hp_space)
    for info on additional available pls arguments.
    """
    def _name(msg):
        return f"{name}.cca_{msg}"

    hp_space = _pls_hp_space(_name, **kwargs)

    return scope.sklearn_CCA(**hp_space)


@validate(params=["algorithm"],
          validation_test=lambda param: isinstance(param, str) and param in ["nipals", "svd"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['nipals', 'svd'].")
def pls_canonical(name: str, algorithm: str = None, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.cross_decomposition.PLSCanonical model.

    Args:
        name: name | str
        algorithm: algorithm for first singular vectors | str

    See help(hpsklearn.components.cross_decomposition._pls._pls_hp_space)
    for info on additional available pls arguments.
    """
    def _name(msg):
        return f"{name}.pls_canonical_{msg}"

    hp_space = _pls_hp_space(_name, **kwargs)
    hp_space["algorithm"] = algorithm or hp.choice(_name("algorithm"), ["nipals", "svd"])

    return scope.sklearn_PLSCanonical(**hp_space)


def pls_regression(name: str, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.cross_decomposition.PLSRegression model.

    Args:
        name: name | str

    See help(hpsklearn.components.cross_decomposition._pls._pls_hp_space)
    for info on additional available pls arguments.
    """
    def _name(msg):
        return f"{name}.pls_regression_{msg}"

    hp_space = _pls_hp_space(_name, **kwargs)

    return scope.sklearn_PLSRegression(**hp_space)

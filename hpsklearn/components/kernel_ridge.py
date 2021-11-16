from hpsklearn.components._base import validate

from hyperopt.pyll import scope
from hyperopt import hp

from sklearn import kernel_ridge
import numpy.typing as npt
import typing


@scope.define
def sklearn_KernelRidge(*args, **kwargs):
    return kernel_ridge.KernelRidge(*args, **kwargs)


@validate(params=["kernel"],
          validation_test=lambda param: isinstance(param, str) and
                                        param in ["additive_chi2", "chi2", "cosine", "linear", "poly",
                                                  "polynomial", "rbf", "laplacian", "sigmoid"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['additive_chi2', 'chi2', 'cosine', 'linear', "
              "'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid'].")
def hp_sklearn_kernel_ridge(name: str,
                            alpha: typing.Union[float, npt.ArrayLike] = None,
                            kernel: typing.Union[str, callable] = None,
                            gamma: float = None,
                            degree: float = None,
                            coef0: float = None,
                            kernel_params: map = None):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.kernel_ridge.KernelRidge model.

    Args:
        name: name | str
        alpha: regularization strength | float, ArrayLike
        kernel: kernel mapping | str, callable
        gamma: gamma parameter specific kernels | float
        degree: degree of polynomial kernel | float
        coef0: zero coefficient for poly and sigmoid kernel | float
        kernel_params: additional kernel parameters | map
    """
    def _name(msg):
        return f"{name}.kernel_ridge_{msg}"

    hp_space = dict(
        alpha=hp.uniform(_name("alpha"), 0.0, 1.0) if alpha is None else alpha,
        kernel=kernel or hp.choice(_name("kernel"),
                                   ["additive_chi2", "chi2", "cosine", "linear", "poly",
                                    "polynomial", "rbf", "laplacian", "sigmoid"]),
        gamma=hp.uniform(_name("gamma"), 0.0, 1.0) if gamma is None else gamma,
        degree=degree or scope.int(hp.uniform(_name("degree"), 1, 7)),
        coef0=hp.uniform(_name("coef0"), 0.0, 1.0) if coef0 is None else coef0,
        kernel_params=kernel_params,
    )
    return scope.sklearn_KernelRidge(**hp_space)

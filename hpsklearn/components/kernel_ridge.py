from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import kernel_ridge
import numpy.typing as npt
import typing


@scope.define
def sklearn_KernelRidge(*args, **kwargs):
    return kernel_ridge.KernelRidge(*args, **kwargs)


@validate(params=["kernel"],
          validation_test=lambda param: not isinstance(param, str) or
                                        param in ["additive_chi2", "chi2", "cosine", "linear", "poly",  # noqa
                                                  "polynomial", "rbf", "laplacian", "sigmoid"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['additive_chi2', 'chi2', 'cosine', 'linear', "
              "'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid'].")
def hp_sklearn_kernel_ridge(name: str,
                            alpha: typing.Union[float, npt.ArrayLike, Apply] = None,
                            kernel: typing.Union[str, callable, Apply] = None,
                            gamma: typing.Union[float, Apply] = None,
                            degree: typing.Union[float, Apply] = None,
                            coef0: typing.Union[float, Apply] = None,
                            kernel_params: map = None,
                            **kwargs):
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
        kernel=hp.choice(_name("kernel"), ["additive_chi2", "chi2", "cosine", "linear", "poly", "polynomial", "rbf",
                                           "laplacian", "sigmoid"]) if kernel is None else kernel,
        gamma=hp.uniform(_name("gamma"), 0.0, 1.0) if gamma is None else gamma,
        degree=scope.int(hp.uniform(_name("degree"), 1, 7)) if degree is None else degree,
        coef0=hp.uniform(_name("coef0"), 0.0, 1.0) if coef0 is None else coef0,
        kernel_params=kernel_params,
        **kwargs
    )
    return scope.sklearn_KernelRidge(**hp_space)

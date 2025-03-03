from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import linear_model
import numpy as np
import typing


@scope.define
def sklearn_OrthogonalMatchingPursuit(*args, **kwargs):
    return linear_model.OrthogonalMatchingPursuit(*args, **kwargs)


@scope.define
def sklearn_OrthogonalMatchingPursuitCV(*args, **kwargs):
    return linear_model.OrthogonalMatchingPursuitCV(*args, **kwargs)


def orthogonal_matching_pursuit(name: str,
                                n_nonzero_coefs: int = None,
                                tol: typing.Union[float, Apply] = None,
                                fit_intercept: bool = True,
                                precompute: typing.Union[str, bool] = "auto",
                                **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.OrthogonalMatchingPursuit model.

    Args:
        name: name | str
        n_nonzero_coefs: target number non-zero coefficients | int
        tol: maximum norm of residual | float
        fit_intercept: whether to calculate intercept for model | bool
        precompute: whether to use precomputed Gram and Xy matrix | str, bool
    """

    def _name(msg):
        return f"{name}.orthogonal_matching_pursuit_{msg}"

    hp_space = dict(
        n_nonzero_coefs=n_nonzero_coefs,
        tol=hp.loguniform(_name("tol"), np.log(1e-5), np.log(1e-2)) if tol is None else tol,
        fit_intercept=fit_intercept,
        precompute=precompute,
        **kwargs
    )

    return scope.sklearn_OrthogonalMatchingPursuit(**hp_space)


def orthogonal_matching_pursuit_cv(name: str,
                                   copy: bool = True,
                                   fit_intercept: bool = True,
                                   max_iter: typing.Union[int, Apply] = None,
                                   cv: typing.Union[int, callable, typing.Generator, Apply] = None,
                                   n_jobs: int = 1,
                                   verbose: typing.Union[bool, int] = False,
                                   **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.OrthogonalMatchingPursuitCV model.

    Args:
        name: name | str
        copy: whether design matrix must be copied | bool
        fit_intercept: whether to calculate intercept for model | bool
        max_iter: maximum number of iterations | int
        cv: cross-validation splitting strategy| int, callable or generator
        n_jobs: number of CPUs during cv | int
        verbose: verbosity amount | bool, int
    """

    def _name(msg):
        return f"{name}.orthogonal_matching_pursuit_cv_{msg}"

    hp_space = dict(
        copy=copy,
        fit_intercept=fit_intercept,
        max_iter=max_iter,
        cv=hp.pchoice(_name("cv"), [(0.0625, 3), (0.175, 4), (0.525, 5), (0.175, 6), (0.0625, 7)])
        if cv is None else cv,
        n_jobs=n_jobs,
        verbose=verbose,
        **kwargs
    )

    return scope.sklearn_OrthogonalMatchingPursuitCV(**hp_space)

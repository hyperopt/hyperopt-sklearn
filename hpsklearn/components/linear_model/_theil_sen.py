from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import linear_model
import numpy as np
import typing


@scope.define
def sklearn_TheilSenRegressor(*args, **kwargs):
    return linear_model.TheilSenRegressor(*args, **kwargs)


@validate(params=["max_iter"],
          validation_test=lambda param: not isinstance(param, int) or param > 0,
          msg="Invalid parameter '%s' with value '%s'. Value must be a positive integer.")
def theil_sen_regressor(
        name: str,
        fit_intercept: bool = True,
        max_subpopulation: typing.Union[int, Apply] = None,
        n_subsamples: int = None,
        max_iter: typing.Union[int, Apply] = None,
        tol: typing.Union[float, Apply] = None,
        random_state=None,
        n_jobs: int = 1,
        verbose: bool = False,
        **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.TheilSenRegressor model.

    Args:
        name: name | str
        fit_intercept: whether to calculate the intercept | bool
        max_subpopulation: consider stochastic subpopulation | int
        n_subsamples: number of samples to calculate parameters | int
        max_iter: maximum number of iterations | int
        tol: tolerance when calculating spatial median | float
        random_state: random state | int
        n_jobs: number of CPUs to use | int
        verbose: verbosity level | bool
    """

    def _name(msg):
        return f"{name}.theil_sen_regressor_{msg}"

    hp_space = dict(
        fit_intercept=fit_intercept,
        max_subpopulation=scope.int(hp.uniform(_name("max_subpopulation"), 7500, 12500))
        if max_subpopulation is None else max_subpopulation,
        n_subsamples=n_subsamples,
        max_iter=scope.int(hp.uniform(_name("max_iter"), 200, 400)) if max_iter is None else max_iter,
        tol=hp.loguniform(_name("tol"), np.log(1e-5), np.log(1e-2)) if tol is None else tol,
        random_state=hp.randint(_name("random_state"), 5) if random_state is None else random_state,
        n_jobs=n_jobs,
        verbose=verbose,
        **kwargs
    )
    return scope.sklearn_TheilSenRegressor(**hp_space)

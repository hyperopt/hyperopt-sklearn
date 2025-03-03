from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import gaussian_process
import numpy as np
import typing


@scope.define
def sklearn_GaussianProcessRegressor(*args, **kwargs):
    return gaussian_process.GaussianProcessRegressor(*args, **kwargs)


@validate(params=["optimizer"],
          validation_test=lambda param: not isinstance(param, str) or param == "fmin_l_bfgs_b",
          msg="Invalid parameter '%s' with value '%s'. Value must be 'fmin_l_bfgs_b' or callable.")
def gaussian_process_regressor(name: str,
                               kernel=None,
                               alpha: typing.Union[float, np.ndarray, Apply] = None,
                               optimizer: typing.Union[str, callable, Apply] = None,
                               n_restarts_optimizer: typing.Union[int, Apply] = None,
                               normalize_y: bool = False,
                               copy_X_train: bool = True,
                               random_state=None,
                               **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.gaussian_process.GaussianProcessRegressor model.

    Args:
        name: name | str
        kernel: kernel instance
        alpha: add value to diagonal of kernel matrix during fitting | float, np.ndarray
        optimizer: optimizer for kernel parameter optimization | str, callable
        n_restarts_optimizer: number of restarts optimizer | int
        normalize_y: normalize target values | bool
        copy_X_train: store persistent copy of training data | bool
        random_state: random seed for center initialization | int
    """

    def _name(msg):
        return f"{name}.gaussian_process_regressor_{msg}"

    hp_space = dict(
        kernel=kernel,
        alpha=hp.loguniform(_name("alpha"), np.log(1e-10), np.log(1e-2)) if alpha is None else alpha,
        optimizer="fmin_l_bfgs_b" if optimizer is None else optimizer,
        n_restarts_optimizer=hp.pchoice(_name("n_restarts_optimizer"), [(0.5, 0), (0.10, 1), (0.10, 2), (0.10, 3),
                                                                        (0.10, 4), (0.10, 5)])
        if n_restarts_optimizer is None else n_restarts_optimizer,
        normalize_y=normalize_y,
        copy_X_train=copy_X_train,
        random_state=hp.randint(_name("random_state"), 5) if random_state is None else random_state,
        **kwargs
    )
    return scope.sklearn_GaussianProcessRegressor(**hp_space)

from hpsklearn.components._base import validate

from hyperopt.pyll import scope
from hyperopt import hp

from sklearn import linear_model
import numpy as np
import typing


@scope.define
def sklearn_Perceptron(*args, **kwargs):
    return linear_model.Perceptron(*args, **kwargs)


@validate(params=["penalty"],
          validation_test=lambda param: isinstance(param, str) and param in ["l1", "l2", "elasticnet"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['l1', 'l2', 'elasticnet'].")
def perceptron(name: str,
               penalty: str = None,
               alpha: float = None,
               l1_ratio: float = None,
               fit_intercept: bool = True,
               max_iter: int = None,
               tol: float = None,
               shuffle: bool = True,
               verbose: int = 0,
               eta0: float = None,
               n_jobs: int = 1,
               random_state=None,
               early_stopping: bool = False,
               validation_fraction: float = None,
               n_iter_no_change: int = 5,
               class_weight: typing.Union[dict, str] = None,
               warm_start: bool = False
               ):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.Perceptron model.

    Args:
        name: name | str
        penalty: penalty or regularization term to use | str
        alpha: constant to multiply regularization term | float
        l1_ratio: elastic net mixing parameter | float
        fit_intercept: whether to estimate intercept | bool
        max_iter: maximum epochs or iterations | int
        tol: stopping criterion | float
        shuffle: shuffle training data after each epoch | bool
        verbose: verbosity level | int
        eta0: constant by which updates are multiplied | float
        n_jobs: number of CPUs to use | int
        random_state: used to shuffle training data | int
        early_stopping: whether to use early stopping | bool
        validation_fraction: validation set for early stopping | float
        n_iter_no_change: iterations with no improvement | int
        class_weight: class_weight fit parameters | dict or str
        warm_start: reuse previous call as fit | bool
    """
    def _name(msg):
        return f"{name}.perceptron_{msg}"

    hp_space = dict(
        penalty=penalty or hp.choice(_name("penalty"), ["l1", "l2", "elasticnet"]),
        alpha=alpha or hp.loguniform(_name("alpha"), np.log(1e-6), np.log(1e-1)),
        l1_ratio=l1_ratio or hp.loguniform(_name("l1_ratio"), np.log(1e-7), np.log(1)),
        fit_intercept=fit_intercept or hp.choice(_name("fit_intercept"), [True, False]),
        max_iter=max_iter or hp.qloguniform(_name("max_iter"), np.log(750), np.log(1250), 1),
        tol=tol or hp.loguniform(_name("tol"), np.log(1e-5), np.log(1e-2)),
        shuffle=shuffle,
        verbose=verbose,
        eta0=eta0 or hp.normal(_name("eta0"), mu=1.0, sigma=0.1),
        n_jobs=n_jobs,
        random_state=hp.randint(_name("random_state"), 5) if random_state is None else random_state,
        early_stopping=early_stopping,
        validation_fraction=0.1 if validation_fraction is None else validation_fraction,
        n_iter_no_change=n_iter_no_change or hp.pchoice(_name("n_iter_no_change"), [(0.25, 4), (0.50, 5), (0.25, 6)]),
        class_weight=class_weight,
        warm_start=warm_start
    )

    return scope.sklearn_Perceptron(**hp_space)

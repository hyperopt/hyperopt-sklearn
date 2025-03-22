from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import linear_model
import numpy.typing as npt
import numpy as np
import typing


@scope.define
def sklearn_Lars(*args, **kwargs):
    return linear_model.Lars(*args, **kwargs)


@scope.define
def sklearn_LassoLars(*args, **kwargs):
    return linear_model.LassoLars(*args, **kwargs)


@scope.define
def sklearn_LarsCV(*args, **kwargs):
    return linear_model.LarsCV(*args, **kwargs)


@scope.define
def sklearn_LassoLarsCV(*args, **kwargs):
    return linear_model.LassoLarsCV(*args, **kwargs)


@scope.define
def sklearn_LassoLarsIC(*args, **kwargs):
    return linear_model.LassoLarsIC(*args, **kwargs)


def _least_angle_n_nonzero_coefs(name: str):
    """
    Declaration search space 'n_nonzero_coefs' parameter
    """
    return scope.int(hp.qloguniform(name, low=np.log(400), high=np.log(600), q=1.0))


def _least_angle_alpha(name: str):
    """
    Declaration search space 'alpha' parameter
    """
    return hp.normal(name, mu=1.0, sigma=.1)


def _least_angle_max_iter(name: str):
    """
    Declaration search space 'max_iter' parameter
    """
    return scope.int(hp.uniform(name, low=250, high=750))


def _least_angle_cv(name: str):
    """
    Declaration search space 'cv' parameter
    """
    return hp.pchoice(name, [
        # create custom distribution
        (0.0625, 3),
        (0.175, 4),
        (0.525, 5),
        (0.175, 6),
        (0.0625, 7),
    ])


def _least_angle_criterion(name: str):
    """
    Declaration search space 'criterion' parameter
    """
    return hp.choice(name, ["bic", "aic"])


def _least_angle_max_n_alphas(name: str):
    """
    Declaration search space 'max_n_alphas' parameter
    """
    return scope.int(hp.loguniform(name, low=np.log(750), high=np.log(1250)))


def _least_angle_random_state(name: str):
    """
    Declaration search space 'random_state' parameter
    """
    return hp.randint(name, 5)


def _least_angle_hp_space(
        name_func,
        fit_intercept: bool = True,
        verbose: typing.Union[bool, int] = False,
        precompute: typing.Union[bool, str, npt.ArrayLike] = "auto",
        eps: float = np.finfo(float).eps,
        copy_X: bool = True,
        **kwargs,
):
    """
    Hyper parameter of search space of common parameters for
     lars
     lasso lars
     lars cv
     lasso lars cv
     lasso lars ic
    """
    hp_space = dict(
        fit_intercept=fit_intercept,
        verbose=verbose,
        precompute=precompute,
        eps=eps,
        copy_X=copy_X,
        **kwargs
    )
    return hp_space


@validate(params=["cv"],
          validation_test=lambda param: not isinstance(param, float) or param > 0,
          msg="Invalid parameter '%s' with value '%s'. Parameter value must exceed 0.")
def _least_angle_cv_shared_space(
        hp_space: dict,
        _name: callable,
        max_iter: typing.Union[int, Apply] = None,
        cv: typing.Union[int, callable, typing.Generator, Apply] = None,
        max_n_alphas: typing.Union[int, Apply] = None,
        n_jobs: int = None,
        **kwargs):
    """
    Declaration shared search space parameters for
     lars cv
     lasso lars cv
    """
    hp_space["max_iter"] = _least_angle_max_iter(_name("max_iter")) if max_iter is None else max_iter
    hp_space["cv"] = _least_angle_cv(_name("cv")) if cv is None else cv
    hp_space["max_n_alphas"] = _least_angle_max_n_alphas(_name("max_n_alphas")) \
        if max_n_alphas is None else max_n_alphas
    hp_space["n_jobs"] = n_jobs
    return hp_space


@validate(params=["n_nonzero_coefs"],
          validation_test=lambda param: not isinstance(param, int) or param > 1,
          msg="Invalid parameter '%s' with value '%s'. Parameter value must exceed 1.")
def lars(name: str,
         n_nonzero_coefs: typing.Union[int, Apply] = None,
         fit_path: bool = True,
         jitter: float = None,
         random_state=None,
         **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.Lars model.

    Args:
        name: name | str
        n_nonzero_coefs: target number non-zero coefficients | int
        fit_path: store full path | bool
        jitter: upper bound noise parameter for stability | float
        random_state: random number generation for jitter

    See help(hpsklearn.components.linear_model._least_angle._least_angle_hp_space)
    for info on additional available least angle arguments.
    """

    def _name(msg):
        return f"{name}.lars_{msg}"

    hp_space = _least_angle_hp_space(_name, **kwargs)
    hp_space["n_nonzero_coefs"] = _least_angle_n_nonzero_coefs(_name("n_nonzero_coefs")) \
        if n_nonzero_coefs is None else n_nonzero_coefs
    hp_space["fit_path"] = fit_path
    hp_space["jitter"] = jitter  # highly dependent on data
    hp_space["random_state"] = _least_angle_random_state(_name("random_state")) \
        if random_state is None else random_state

    return scope.sklearn_Lars(**hp_space)


@validate(params=["alpha"],
          validation_test=lambda param: not isinstance(param, int) or param > 1,
          msg="Invalid parameter '%s' with value '%s'. Alpha = 0 is equivalent to ordinary least square. "
              "For ordinary least square, use LinearRegression instead.")
@validate(params=["max_iter"],
          validation_test=lambda param: not isinstance(param, int) or param > 1,
          msg="Invalid parameter '%s' with value '%s'. Parameter value must exceed 1.")
def lasso_lars(name: str,
               alpha: typing.Union[float, Apply] = None,
               max_iter: typing.Union[int, Apply] = None,
               fit_path: bool = True,
               positive: bool = False,
               jitter: float = None,
               random_state=None,
               **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.LassoLars model.

    Args:
        name: name | str
        alpha: constant that multiplies penalty | float
        max_iter: maximum number of iterations | int
        fit_path: store full path | bool
        positive: restrict coefficient to positive | bool
        jitter: upper bound noise parameter for stability | float
        random_state: random number generation for jitter

    See help(hpsklearn.components.linear_model._least_angle._least_angle_hp_space)
    for info on additional available least angle arguments.
    """

    def _name(msg):
        return f"{name}.lasso_lars_{msg}"

    hp_space = _least_angle_hp_space(_name, **kwargs)
    hp_space["alpha"] = _least_angle_alpha(_name("alpha")) if alpha is None else alpha
    hp_space["max_iter"] = _least_angle_max_iter(_name("max_iter")) if max_iter is None else max_iter
    hp_space["fit_path"] = fit_path
    hp_space["positive"] = positive
    hp_space["jitter"] = jitter  # highly dependent on data
    hp_space["random_state"] = _least_angle_random_state(_name("random_state")) \
        if random_state is None else random_state

    return scope.sklearn_LassoLars(**hp_space)


@validate(params=["max_iter", "max_n_alphas"],
          validation_test=lambda param: not isinstance(param, int) or param > 0,
          msg="Invalid parameter '%s' with value '%s'. Parameter value must exceed 0.")
def lars_cv(name: str,
            max_iter: typing.Union[int, Apply] = None,
            cv: typing.Union[int, callable, typing.Generator, Apply] = None,
            max_n_alphas: typing.Union[int, Apply] = None,
            n_jobs: int = None,
            **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.LarsCV model.

    Args:
        name: name | str
        max_iter: maximum number of iterations | int
        cv: cross-validation splitting strategy | int, callable or generator
        max_n_alphas: max number of points to compute residuals | int
        n_jobs: number of CPUs to use during CV | int


    See help(hpsklearn.components.linear_model._least_angle._least_angle_hp_space)
    for info on additional available least angle arguments.
    """

    def _name(msg):
        return f"{name}.lars_cv_{msg}"

    hp_space = _least_angle_hp_space(_name, **kwargs)
    hp_space = _least_angle_cv_shared_space(hp_space=hp_space, _name=_name,
                                            max_iter=max_iter, cv=cv,
                                            max_n_alphas=max_n_alphas,
                                            n_jobs=n_jobs)

    return scope.sklearn_LarsCV(**hp_space)


@validate(params=["max_iter", "max_n_alphas"],
          validation_test=lambda param: not isinstance(param, int) or param > 0,
          msg="Invalid parameter '%s' with value '%s'. Parameter value must exceed 0.")
def lasso_lars_cv(name: str,
                  max_iter: typing.Union[int, Apply] = None,
                  cv: typing.Union[int, callable, typing.Generator, Apply] = None,
                  max_n_alphas: typing.Union[int, Apply] = None,
                  n_jobs: int = None,
                  positive: bool = False,
                  **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.LassoLarsCV model.

    Args:
        name: name | str
        max_iter: maximum number of iterations | int
        cv: cross-validation splitting strategy | int, callable or generator
        max_n_alphas: max number of points to compute residuals | int
        n_jobs: number of CPUs to use during CV | int
        positive: restrict coefficient to positive | bool

    See help(hpsklearn.components.linear_model._least_angle._least_angle_hp_space)
    for info on additional available least angle arguments.
    """

    def _name(msg):
        return f"{name}.lasso_lars_cv_{msg}"

    hp_space = _least_angle_hp_space(_name, **kwargs)
    hp_space = _least_angle_cv_shared_space(hp_space=hp_space, _name=_name,
                                            max_iter=max_iter, cv=cv,
                                            max_n_alphas=max_n_alphas,
                                            n_jobs=n_jobs)
    hp_space["positive"] = positive

    return scope.sklearn_LassoLarsCV(**hp_space)


@validate(params=["criterion"],
          validation_test=lambda param: not isinstance(param, str) or param in ("bic", "aic"),
          msg="Invalid parameter '%s' with value '%s'. Choose 'bic' or 'aic'.'")
@validate(params=["max_iter"],
          validation_test=lambda param: not isinstance(param, int) or param > 0,
          msg="Invalid parameter '%s' with value '%s'. Parameter value must exceed 0.")
def lasso_lars_ic(name: str,
                  criterion: typing.Union[str, Apply] = None,
                  max_iter: typing.Union[int, Apply] = None,
                  positive: bool = False,
                  **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.LassoLarsIC model.

    Args:
        name: name | str
        criterion: choose from 'bic', 'aic' | str
        max_iter: maximum number of iterations | int
        positive: restrict coefficient to positive | bool

    See help(hpsklearn.components.linear_model._least_angle._least_angle_hp_space)
    for info on additional available least angle arguments.
    """

    def _name(msg):
        return f"{name}.lasso_lars_ic_{msg}"

    hp_space = _least_angle_hp_space(_name, **kwargs)
    hp_space["criterion"] = _least_angle_criterion(_name("criterion")) if criterion is None else criterion
    hp_space["max_iter"] = _least_angle_max_iter(_name("max_iter")) if max_iter is None else max_iter
    hp_space["positive"] = positive

    return scope.sklearn_LassoLarsIC(**hp_space)

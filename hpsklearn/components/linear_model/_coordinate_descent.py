from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import linear_model
import numpy.typing as npt
import numpy as np
import typing


@scope.define
def sklearn_Lasso(*args, **kwargs):
    return linear_model.Lasso(*args, **kwargs)


@scope.define
def sklearn_ElasticNet(*args, **kwargs):
    return linear_model.ElasticNet(*args, **kwargs)


@scope.define
def sklearn_LassoCV(*args, **kwargs):
    return linear_model.LassoCV(*args, **kwargs)


@scope.define
def sklearn_ElasticNetCV(*args, **kwargs):
    return linear_model.ElasticNetCV(*args, **kwargs)


@scope.define
def sklearn_MultiTaskLasso(*args, **kwargs):
    return linear_model.MultiTaskLasso(*args, **kwargs)


@scope.define
def sklearn_MultiTaskElasticNet(*args, **kwargs):
    return linear_model.MultiTaskElasticNet(*args, **kwargs)


@scope.define
def sklearn_MultiTaskLassoCV(*args, **kwargs):
    return linear_model.MultiTaskLassoCV(*args, **kwargs)


@scope.define
def sklearn_MultiTaskElasticNetCV(*args, **kwargs):
    return linear_model.MultiTaskElasticNetCV(*args, **kwargs)


def _coordinate_descent_alpha(name: str):
    """
    Declaration search space 'alpha' parameter
    """
    return hp.loguniform(name, low=np.log(1e-6), high=np.log(1e-1))


def _coordinate_descent_l1_ratio(name: str):
    """
    Declaration search space 'l1_ratio'  parameter
    """
    return hp.uniform(name, low=0, high=1)


def _coordinate_descent_eps(name: str):
    """
    Declaration search space 'eps' parameter
    """
    return hp.lognormal(name, mu=np.log(1e-3), sigma=np.log(10))


def _coordinate_descent_n_alphas(name: str):
    """
    Declaration search space 'n_alphas' parameter
    """
    return scope.int(hp.uniform(name, low=80, high=120))


def _coordinate_descent_max_iter(name: str):
    """
    Declaration search space 'max_iter' parameter
    """
    return scope.int(hp.uniform(name, low=800, high=1200))


def _coordinate_descent_tol(name: str):
    """
    Declaration search space 'tol' parameter
    """
    return hp.loguniform(name, low=np.log(1e-5), high=np.log(1e-2))


def _coordinate_descent_cv(name: str):
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


def _coordinate_descent_random_state(name: str):
    """
    Declaration search space 'random_state' parameter
    """
    return hp.randint(name, 5)


def _coordinate_descent_selection(name: str):
    """
    Declaration search space 'selection' parameter
    """
    return hp.choice(name, ["cyclic", "random"])


@validate(params=["alpha"],
          validation_test=lambda param: not isinstance(param, int) or param > 0,
          msg="Invalid parameter '%s' with value '%s'. Alpha = 0 is equivalent to ordinary least square. "
              "For ordinary least square, use LinearRegression instead.")
@validate(params=["selection"],
          validation_test=lambda param: not isinstance(param, str) or param in ["cyclic", "random"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['cyclic', 'random'].")
@validate(params=["max_iter"],
          validation_test=lambda param: not isinstance(param, int) or param > 0,
          msg="Invalid parameter '%s' with value '%s'. Parameter value must exceed 0.")
def _coordinate_descent_hp_space(
        name_func,
        alpha: typing.Union[float, Apply] = None,
        fit_intercept: bool = True,
        copy_X: bool = True,
        max_iter: typing.Union[int, Apply] = None,
        tol: typing.Union[float, Apply] = None,
        warm_start: bool = False,
        random_state=None,
        selection: typing.Union[str, Apply] = None,
        **kwargs
):
    """
    Hyper parameter search space for
     lasso
     elastic net
     multi task lasso
     multi task elastic net
    """
    hp_space = dict(
        alpha=_coordinate_descent_alpha(name_func("alpha")) if alpha is None else alpha,
        fit_intercept=fit_intercept,
        copy_X=copy_X,
        max_iter=_coordinate_descent_max_iter(name_func("max_iter")) if max_iter is None else max_iter,
        tol=_coordinate_descent_tol(name_func("tol")) if tol is None else tol,
        warm_start=warm_start,
        random_state=_coordinate_descent_random_state(name_func("random_state"))
        if random_state is None else random_state,
        selection=_coordinate_descent_selection(name_func("selection")) if selection is None else selection,
        **kwargs
    )
    return hp_space


@validate(params=["selection"],
          validation_test=lambda param: not isinstance(param, str) or param in ["cyclic", "random"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['cyclic', 'random'].")
@validate(params=["max_iter", "n_alphas", "eps", "cv"],
          validation_test=lambda param: not isinstance(param, float) or param > 0,
          msg="Invalid parameter '%s' with value '%s'. Parameter value must exceed 0.")
def _coordinate_descent_cv_hp_space(
        name_func,
        eps: typing.Union[float, Apply] = None,
        n_alphas: typing.Union[int, Apply] = None,
        alphas: np.ndarray = None,
        fit_intercept: bool = True,
        max_iter: typing.Union[int, Apply] = None,
        tol: typing.Union[float, Apply] = None,
        copy_X: bool = True,
        cv: typing.Union[int, callable, typing.Generator, Apply] = None,
        verbose: int = False,
        n_jobs: int = 1,
        random_state=None,
        selection: typing.Union[str, Apply] = None,
        **kwargs
):
    """
    Hyper parameter search space for
     lasso cv
     elastic net cv
     multi task lasso cv
     multi task elastic net cv
    """
    hp_space = dict(
        eps=_coordinate_descent_eps(name_func("eps")) if eps is None else eps,
        n_alphas=_coordinate_descent_n_alphas(name_func("n_alphas")) if n_alphas is None else n_alphas,
        alphas=alphas,
        fit_intercept=fit_intercept,
        max_iter=_coordinate_descent_max_iter(name_func("max_iter")) if max_iter is None else max_iter,
        tol=_coordinate_descent_tol(name_func("tol")) if tol is None else tol,
        copy_X=copy_X,
        cv=_coordinate_descent_cv(name_func("cv")) if cv is None else cv,
        verbose=verbose,
        n_jobs=n_jobs,
        random_state=_coordinate_descent_random_state(name_func("random_state"))
        if random_state is None else random_state,
        selection=_coordinate_descent_selection(name_func("selection")) if selection is None else selection,
        **kwargs
    )
    return hp_space


def lasso(name: str,
          precompute: typing.Union[bool, npt.ArrayLike] = False,
          positive: bool = False,
          **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.Lasso model.

    Args:
        name: name | str
        precompute: use precomputed Gram matrix | bool, str or Array
        positive: restrict coefficient to positive | bool

    See help(hpsklearn.components.linear_model._coordinate_descent._coordinate_descent_hp_space)
    for info on additional available coordinate descent arguments.
    """

    def _name(msg):
        return f"{name}.lasso_{msg}"

    hp_space = _coordinate_descent_hp_space(_name, **kwargs)
    hp_space["precompute"] = precompute
    hp_space["positive"] = positive

    return scope.sklearn_Lasso(**hp_space)


def elastic_net(name: str,
                l1_ratio: typing.Union[float, Apply] = None,
                precompute: typing.Union[bool, npt.ArrayLike] = False,
                positive: bool = False,
                **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.ElasticNet model.

    Args:
        name: name | str
        l1_ratio: mixing parameter | float
        precompute: use precomputed Gram matrix | bool, str or Array
        positive: restrict coefficient to positive | bool

    See help(hpsklearn.components.linear_model._coordinate_descent._coordinate_descent_hp_space)
    for info on additional available coordinate descent arguments.
    """

    def _name(msg):
        return f"{name}.elastic_net_{msg}"

    hp_space = _coordinate_descent_hp_space(_name, **kwargs)
    hp_space["l1_ratio"] = _coordinate_descent_l1_ratio(_name("l1_ratio")) if l1_ratio is None else l1_ratio
    hp_space["precompute"] = precompute
    hp_space["positive"] = positive

    return scope.sklearn_ElasticNet(**hp_space)


def lasso_cv(name: str,
             precompute: typing.Union[bool, str, npt.ArrayLike] = "auto",
             positive: bool = False,
             **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.LassoCV model.

    Args:
        name: name | str
        precompute: use precomputed Gram matrix | bool, str or Array
        positive: restrict coefficient to positive | bool

    See help(hpsklearn.components.linear_model._coordinate_descent._coordinate_descent_cv_hp_space)
    for info on additional available coordinate descent cv arguments.
    """

    def _name(msg):
        return f"{name}.lasso_cv_{msg}"

    hp_space = _coordinate_descent_cv_hp_space(_name, **kwargs)
    hp_space["precompute"] = precompute
    hp_space["positive"] = positive

    return scope.sklearn_LassoCV(**hp_space)


def elastic_net_cv(name: str,
                   l1_ratio: typing.Union[float, Apply] = None,
                   precompute: typing.Union[bool, str, npt.ArrayLike] = "auto",
                   positive: bool = False,
                   **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.ElasticNetCV model.

    Args:
        name: name | str
        l1_ratio: mixing parameter | float
        precompute: use precomputed Gram matrix | bool, str or Array
        positive: restrict coefficient to positive | bool

    See help(hpsklearn.components.linear_model._coordinate_descent._coordinate_descent_cv_hp_space)
    for info on additional available coordinate descent cv arguments.
    """

    def _name(msg):
        return f"{name}.elastic_net_cv_{msg}"

    hp_space = _coordinate_descent_cv_hp_space(_name, **kwargs)
    hp_space["l1_ratio"] = _coordinate_descent_l1_ratio(_name("l1_ratio")) if l1_ratio is None else l1_ratio
    hp_space["precompute"] = precompute
    hp_space["positive"] = positive

    return scope.sklearn_ElasticNetCV(**hp_space)


def multi_task_lasso(name: str, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.MultiTaskLasso model.

    Args:
        name: name | str

    See help(hpsklearn.components.linear_model._coordinate_descent._coordinate_descent_hp_space)
    for info on additional available coordinate descent arguments.
    """

    def _name(msg):
        return f"{name}.multi_task_lasso_{msg}"

    hp_space = _coordinate_descent_hp_space(_name, **kwargs)

    return scope.sklearn_MultiTaskLasso(**hp_space)


def multi_task_elastic_net(name: str, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.MultiTaskElasticNet model.

    Args:
        name: name | str

    See help(hpsklearn.components.linear_model._coordinate_descent._coordinate_descent_hp_space)
    for info on additional available coordinate descent arguments.
    """

    def _name(msg):
        return f"{name}.multi_task_elastic_net_{msg}"

    hp_space = _coordinate_descent_hp_space(_name, **kwargs)

    return scope.sklearn_MultiTaskElasticNet(**hp_space)


def multi_task_lasso_cv(name: str, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.MultiTaskLassoCV model.

    Args:
        name: name | str

    See help(hpsklearn.components.linear_model._coordinate_descent._coordinate_descent_cv_hp_space)
    for info on additional available coordinate descent cv arguments.
    """

    def _name(msg):
        return f"{name}.multi_task_lasso_cv_{msg}"

    hp_space = _coordinate_descent_cv_hp_space(_name, **kwargs)

    return scope.sklearn_MultiTaskLassoCV(**hp_space)


def multi_task_elastic_net_cv(name: str, l1_ratio: typing.Union[float, Apply] = None, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.MultiTaskElasticNetCV model.

    Args:
        name: name | str
        l1_ratio: mixing parameter | float

    See help(hpsklearn.components.linear_model._coordinate_descent._coordinate_descent_cv_hp_space)
    for info on additional available coordinate descent arguments.
    """

    def _name(msg):
        return f"{name}.multi_task_elastic_net_cv_{msg}"

    hp_space = _coordinate_descent_cv_hp_space(_name, **kwargs)
    hp_space["l1_ratio"] = _coordinate_descent_l1_ratio(_name("l1_ratio")) if l1_ratio is None else l1_ratio

    return scope.sklearn_MultiTaskElasticNetCV(**hp_space)

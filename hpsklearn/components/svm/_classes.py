from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import svm
import numpy as np
import typing


@scope.define
def sklearn_LinearSVC(*args, **kwargs):
    return svm.LinearSVC(*args, **kwargs)


@scope.define
def sklearn_LinearSVR(*args, **kwargs):
    return svm.LinearSVR(*args, **kwargs)


@scope.define
def sklearn_NuSVC(*args, **kwargs):
    return svm.NuSVC(*args, **kwargs)


@scope.define
def sklearn_NuSVR(*args, **kwargs):
    return svm.NuSVR(*args, **kwargs)


@scope.define
def sklearn_OneClassSVM(*args, **kwargs):
    return svm.OneClassSVM(*args, **kwargs)


@scope.define
def sklearn_SVC(*args, **kwargs):
    return svm.SVC(*args, **kwargs)


@scope.define
def sklearn_SVR(*args, **kwargs):
    return svm.SVR(*args, **kwargs)


def _linear_tol(name: str):
    """
    Declaration search space 'tol' parameter
    """
    return hp.loguniform(name, np.log(1e-5), np.log(1e-2))


def _linear_C(name: str):
    """
    Declaration search space 'C' parameter
    """
    return hp.uniform(name, 0.5, 1.5)


def _linear_intercept_scaling(name: str):
    """
    Declaration search space 'intercept_scaling' parameter
    """
    return hp.uniform(name, 0.5, 1.5)


def _svm_random_state(name: str):
    """
    Declaration search space 'random_state' parameter
    """
    return hp.randint(name, 5)


def _linear_max_iter(name: str):
    """
    Declaration search space 'max_iter' parameter
    """
    return scope.int(hp.uniform(name, 750, 1500))


def _svm_kernel(name: str):
    """
    Declaration search space 'kernel' parameter
    """
    return hp.choice(name, ["linear", "poly", "rbf", "sigmoid"])


def _svm_degree(name: str):
    """
    Declaration search space 'degree' parameter
    """
    return hp.choice(name, [1, 2, 3, 4, 5])


def _svm_gamma(name: str):
    """
    Declaration search space 'gamma' parameter
    """
    return hp.choice(name, ["auto", "scale"])


def _svm_coef0(name: str):
    """
    Declaration search space 'coef0' parameter
    """
    return hp.uniform(name, 0.0, 1.0)


def _svm_shrinking(name: str):
    """
    Declaration search space 'shrinking' parameter
    """
    return hp.choice(name, [True, False])


def _svm_tol(name: str):
    """
    Declaration search space 'tol' parameter
    """
    return hp.loguniform(name, np.log(1e-5), np.log(1e-2))


def _svc_decision_function_shape(name: str):
    """
    Declaration search space 'decision_function_shape' parameter
    """
    return hp.choice(name, ["ovo", "ovr"])


def _linear_hp_space(
        name_func,
        tol: typing.Union[float, Apply] = None,
        C: typing.Union[float, Apply] = None,
        fit_intercept: bool = True,
        intercept_scaling: typing.Union[float, Apply] = None,
        dual: bool = "auto",
        verbose: int = 0,
        random_state=None,
        max_iter: typing.Union[int, Apply] = None,
        **kwargs
):
    """
    Hyper parameter search space for
     linear svc
     linear svr
    """
    hp_space = dict(
        tol=_linear_tol(name_func("tol")) if tol is None else tol,
        C=_linear_C(name_func("C")) if C is None else C,
        fit_intercept=fit_intercept,
        intercept_scaling=_linear_intercept_scaling(name_func("intercept_scaling"))
        if intercept_scaling is None else intercept_scaling,
        dual=dual,
        verbose=verbose,
        random_state=_svm_random_state(name_func("random_state")) if random_state is None else random_state,
        max_iter=_linear_max_iter(name_func("max_iter")) if max_iter is None else max_iter,
        **kwargs
    )
    return hp_space


@validate(params=["kernel"],
          validation_test=lambda param: not isinstance(param, str) or param in ["linear", "poly", "rbf", "sigmoid",
                                                                                "precomputed"],
          msg="Invalid parameter '%s' with value '%s'. "
              "Value must be in ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'].")
@validate(params=["gamma"],
          validation_test=lambda param: not isinstance(param, str) or param in ["scale", "auto"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['scale', 'auto'].")
@validate(params=["class_weight"],
          validation_test=lambda param: not isinstance(param, str) or param == "balanced",
          msg="Invalid parameter '%s' with value '%s'. Value must be 'balanced'.")
@validate(params=["decision_function_shape"],
          validation_test=lambda param: not isinstance(param, str) or param in ["ovo", "ovr"],
          msg="Invalid parameter '%s' with value '%s'. Value must be ['ovo', 'ovr'].")
def _svc_hp_space(name_func,
                  kernel: typing.Union[str, Apply] = None,
                  degree: typing.Union[int, Apply] = None,
                  gamma: typing.Union[float, str, Apply] = None,
                  coef0: typing.Union[float, Apply] = None,
                  shrinking: typing.Union[bool, Apply] = None,
                  probability: typing.Union[bool, Apply] = False,
                  tol: typing.Union[float, Apply] = None,
                  cache_size: int = 200,
                  class_weight: typing.Union[dict, str] = None,
                  verbose: bool = False,
                  max_iter: typing.Union[int, Apply] = None,
                  decision_function_shape: typing.Union[str, Apply] = None,
                  break_ties: bool = False,
                  random_state=None,
                  **kwargs):
    """
    Hyper parameter search space for
     nu svc
     svc
    """
    hp_space = dict(
        kernel=_svm_kernel(name_func("kernel")) if kernel is None else kernel,
        degree=_svm_degree(name_func("degree")) if degree is None else degree,
        gamma=_svm_gamma(name_func("gamma")) if gamma is None else gamma,
        coef0=_svm_coef0(name_func("coef0")) if coef0 is None else coef0,
        shrinking=_svm_shrinking(name_func("shrinking")) if shrinking is None else shrinking,
        probability=probability,
        tol=_svm_tol(name_func("tol")) if tol is None else tol,
        cache_size=cache_size,
        class_weight=class_weight,
        verbose=verbose,
        max_iter=-1 if max_iter is None else max_iter,
        decision_function_shape=_svc_decision_function_shape(name_func("decision_function_shape"))
        if decision_function_shape is None else decision_function_shape,
        break_ties=break_ties,
        random_state=_svm_random_state(name_func("random_state")) if random_state is None else random_state,
        **kwargs
    )
    return hp_space


def _svr_one_class_hp_space(name_func,
                            kernel: typing.Union[str, Apply] = None,
                            degree: typing.Union[int, Apply] = None,
                            gamma: typing.Union[float, str, Apply] = None,
                            coef0: typing.Union[float, Apply] = None,
                            tol: typing.Union[float, Apply] = None,
                            shrinking: typing.Union[bool, Apply] = None,
                            cache_size: int = 200,
                            verbose: bool = False,
                            max_iter: int = None,
                            **kwargs):
    """
    Hyper parameter search space for
     nu svr
     one class svm
     svr
    """
    hp_space = dict(
        kernel=_svm_kernel(name_func("kernel")) if kernel is None else kernel,
        degree=_svm_degree(name_func("degree")) if degree is None else degree,
        gamma=_svm_gamma(name_func("gamma")) if gamma is None else gamma,
        coef0=_svm_coef0(name_func("coef0")) if coef0 is None else coef0,
        tol=_svm_tol(name_func("tol")) if tol is None else tol,
        shrinking=_svm_shrinking(name_func("shrinking")) if shrinking is None else shrinking,
        cache_size=cache_size,
        verbose=verbose,
        max_iter=-1 if max_iter is None else max_iter,
        **kwargs,
    )
    return hp_space


@validate(params=["penalty"],
          validation_test=lambda param: not isinstance(param, str) or param in ["l1", "l2"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['l1', 'l2'].")
@validate(params=["loss"],
          validation_test=lambda param: not isinstance(param, str) or param in ["hinge", "squared_hinge"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['hinge', 'squared_hinge'].")
@validate(params=["multi_class"],
          validation_test=lambda param: not isinstance(param, str) or param in ["ovr", "crammer_singer"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['ovr', 'crammer_singer'].")
@validate(params=["class_weight"],
          validation_test=lambda param: not isinstance(param, str) or param == "balanced",
          msg="Invalid parameter '%s' with value '%s'. Value must be 'balanced'.")
def linear_svc(name: str,
               penalty: typing.Union[str, Apply] = None,
               loss: typing.Union[str, Apply] = None,
               multi_class: typing.Union[str, Apply] = None,
               class_weight: typing.Union[dict, str] = None,
               **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.svm.LinearSVC model.

    Args:
        name: name | str
        penalty: penalization norm | str
        loss: loss function to use | str
        multi_class: multi-class strategy | str
        class_weight: class_weight fit parameters | dict or str

    See help(hpsklearn.components.svm._classes._linear_hp_space)
    for info on additional available linear svm arguments.
    """

    def _name(msg):
        return f"{name}.linear_svc_{msg}"

    hp_space = _linear_hp_space(_name, **kwargs)
    hp_space["penalty"] = "l2" if penalty is None else penalty
    hp_space["loss"] = "squared_hinge" if loss is None else loss
    hp_space["multi_class"] = hp.choice(_name("multi_class"), ["ovr", "crammer_singer"]) \
        if multi_class is None else multi_class
    hp_space["class_weight"] = class_weight

    return scope.sklearn_LinearSVC(**hp_space)


@validate(params=["loss"],
          validation_test=lambda param: not isinstance(param, str) or param in ["epsilon_insensitive",
                                                                                "squared_epsilon_insensitive"],
          msg="Invalid parameter '%s' with value '%s'. "
              "Value must be in ['epsilon_insensitive', 'squared_epsilon_insensitive'].")
def linear_svr(name: str, epsilon: typing.Union[float, Apply] = None, loss: typing.Union[str, Apply] = None, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.svm.LinearSVR model.

    Args:
        name: name | str
        epsilon: epsilon for loss function | float
        loss: loss function to use | str

    See help(hpsklearn.components.svm._classes._linear_hp_space)
    for info on additional available linear svm arguments.
    """

    def _name(msg):
        return f"{name}.linear_svr_{msg}"

    hp_space = _linear_hp_space(_name, **kwargs)
    hp_space["epsilon"] = 0 if epsilon is None else epsilon
    hp_space["loss"] = hp.choice(_name("loss"), ["epsilon_insensitive", "squared_epsilon_insensitive"]) \
        if loss is None else loss

    return scope.sklearn_LinearSVR(**hp_space)


def nu_svc(name: str, nu: typing.Union[float, Apply] = None, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.svm.NuSVC model.

    Args:
        name: name | str
        nu: upper bound fraction of margin errors | float

    See help(hpsklearn.components.svm._classes._svc_hp_space)
    for info on additional available svc svm arguments.
    """

    def _name(msg):
        return f"{name}.nu_svc_{msg}"

    hp_space = _svc_hp_space(_name, **kwargs)
    hp_space["nu"] = hp.uniform(_name("nu"), 0.01, 0.90) if nu is None else nu

    return scope.sklearn_NuSVC(**hp_space)


def nu_svr(name: str, nu: typing.Union[float, Apply] = None, C: typing.Union[float, Apply] = None, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.svm.NuSVR model.

    Args:
        name: name | str
        nu: upper bound fraction of training errors | float
        C: regularization parameter | float

    See help(hpsklearn.components.svm._classes._svr_one_class_hp_space)
    for info on additional available svr svm arguments.
    """

    def _name(msg):
        return f"{name}.nu_svr_{msg}"

    hp_space = _svr_one_class_hp_space(_name, **kwargs)
    hp_space["nu"] = hp.uniform(_name("nu"), 0.01, 0.90) if nu is None else nu
    hp_space["C"] = hp.normal(_name("C"), 1.0, 0.20) if C is None else C

    return scope.sklearn_NuSVR(**hp_space)


def one_class_svm(name: str, nu: typing.Union[float, Apply] = None, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.svm.OneClassSVM model.

    Args:
        name: name | str
        nu: upper bound fraction of training errors | float

    See help(hpsklearn.components.svm._classes._svr_one_class_hp_space)
    for info on additional available one class svm arguments.
    """

    def _name(msg):
        return f"{name}.one_class_svm_{msg}"

    hp_space = _svr_one_class_hp_space(_name, **kwargs)
    hp_space["nu"] = hp.uniform(_name("nu"), 0.01, 0.90) if nu is None else nu

    return scope.sklearn_OneClassSVM(**hp_space)


def svc(name: str, C: typing.Union[float, Apply] = None, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.svm.SVC model.

    Args:
        name: name | str
        C: regularization parameter | float

    See help(hpsklearn.components.svm._classes._svc_hp_space)
    for info on additional available svc svm arguments.
    """

    def _name(msg):
        return f"{name}.svc_{msg}"

    hp_space = _svc_hp_space(_name, **kwargs)
    hp_space["C"] = hp.normal(_name("C"), 1.0, 0.20) if C is None else C

    return scope.sklearn_SVC(**hp_space)


def svr(name: str, C: typing.Union[float, Apply] = None, epsilon: typing.Union[float, Apply] = None, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.svm.SVR model.

    Args:
        name: name | str
        C: regularization parameter | float
        epsilon: epsilon in epsilon-svr model | float

    See help(hpsklearn.components.svm._classes._svr_one_class_hp_space)
    for info on additional available svr svm arguments.
    """

    def _name(msg):
        return f"{name}.svr_{msg}"

    hp_space = _svr_one_class_hp_space(_name, **kwargs)
    hp_space["C"] = hp.normal(_name("C"), 1.0, 0.20) if C is None else C
    hp_space["epsilon"] = hp.uniform(_name("epsilon"), 0.05, 0.15) if epsilon is None else epsilon

    return scope.sklearn_SVR(**hp_space)

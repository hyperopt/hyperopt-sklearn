from hpsklearn.components._base import validate

from hyperopt.pyll import scope
from hyperopt import hp

from sklearn import linear_model
import numpy as np
import typing


@scope.define
def sklearn_SGDClassifier(*args, **kwargs):
    return linear_model.SGDClassifier(*args, **kwargs)


@scope.define
def sklearn_SGDRegressor(*args, **kwargs):
    return linear_model.SGDRegressor(*args, **kwargs)


@scope.define
def sklearn_SGDOneClassSVM(*args, **kwargs):
    return linear_model.SGDOneClassSVM(*args, **kwargs)


def _stochastic_gradient_classifier_loss(name: str):
    """
    Declaration search space 'loss' parameter
     for stochastic gradient classifier
    """
    return hp.pchoice(name, [
        (0.25, "hinge"),
        (0.25, "log"),
        (0.25, "modified_huber"),
        (0.05, "squared_hinge"),
        (0.05, "perceptron"),
        (0.05, "squared_error"),
        (0.05, "huber"),
        (0.03, "epsilon_insensitive"),
        (0.02, "squared_epsilon_insensitive")
    ])


def _stochastic_gradient_regressor_loss(name: str):
    """
    Declaration search space 'loss' parameter
     for stochastic gradient regressor
    """
    return hp.pchoice(name, [
        (0.33, "squared_error"),
        (0.33, "huber"),
        (0.20, "epsilon_insensitive"),
        (0.14, "squared_epsilon_insensitive")
    ])


def _stochastic_gradient_nu(name: str):
    """
    Declaration search space 'nu' parameter
    """
    return hp.normal(name, mu=0.5, sigma=0.075)


def _stochastic_gradient_penalty(name: str):
    """
    Declaration search space 'penalty' parameter
    """
    return hp.pchoice(name, [
        (0.40, "l2"),
        (0.35, "l1"),
        (0.25, "elasticnet")
    ])


def _stochastic_gradient_alpha(name: str):
    """
    Declaration search space 'alpha' parameter
    """
    return hp.loguniform(name, np.log(1e-6), np.log(1e-1))


def _stochastic_gradient_l1_ratio(name: str):
    """
    Declaration search space 'l1_ratio' parameter
    """
    return hp.loguniform(name, np.log(1e-7), np.log(1))


def _stochastic_gradient_max_iter(name: str):
    """
    Declaration search space 'max_iter' parameter
    """
    return hp.qloguniform(name, np.log(750), np.log(1250), 1)


def _stochastic_gradient_tol(name: str):
    """
    Declaration search space 'tol' parameter
    """
    return hp.loguniform(name, np.log(1e-5), np.log(1e-2))


def _stochastic_gradient_random_state(name: str):
    """
    Declaration search space 'random_state' parameter
    """
    return hp.randint(name, 5)


def _stochastic_gradient_learning_rate(name: str):
    """
    Declaration search space 'learning_rate' parameter
    """
    return hp.pchoice(name, [
        (0.50, "optimal"),
        (0.20, "invscaling"),
        (0.20, "constant"),
        (0.10, "adaptive")
    ])


def _stochastic_gradient_eta0(name: str):
    """
    Declaration search space 'eta0' parameter
    """
    return hp.loguniform(name, np.log(1e-5), np.log(1e-1))


def _stochastic_gradient_power_t(name: str):
    """
    Declaration search space 'power_t' parameter
    """
    return hp.uniform(name, 0, 1)


def _stochastic_gradient_n_iter_no_change(name: str):
    """
    Declaration search space 'n_iter_no_change' parameter
    """
    return hp.pchoice(name, [
        (0.25, 4),
        (0.50, 5),
        (0.25, 6)
    ])


@validate(params=["penalty"],
          validation_test=lambda param: isinstance(param, str) and param in ["l2", "l1", "elasticnet"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['l2', 'l1', 'elasticnet'].")
@validate(params=["learning_rate"],
          validation_test=lambda param: isinstance(param, str) and param in
                                        ["constant", "optimal", "invscaling", "adaptive"],
          msg="Invalid parameter '%s' with value '%s'. "
              "Value must be in ['constant', 'optimal', 'invscaling', 'adaptive'].")
@validate(params=["n_iter_no_change"],
          validation_test=lambda param: isinstance(param, int) and param > 0,
          msg="Invalid parameter '%s' with value '%s'. Value must be 0 or higher.")
def _stochastic_gradient_hp_space(
        name_func,
        loss: str = None,
        penalty: str = None,
        alpha: float = None,
        l1_ratio: float = None,
        fit_intercept: bool = True,
        max_iter: int = None,
        tol: float = None,
        shuffle: bool = True,
        verbose: int = 0,
        epsilon: float = 0.1,
        random_state=None,
        learning_rate: str = None,
        eta0: float = None,
        power_t: float = None,
        early_stopping: bool = False,
        validation_fraction: float = None,
        n_iter_no_change: int = None,
        warm_start: bool = False,
        average: typing.Union[bool, int] = False,
):
    """
    Hyper parameter search space for
     sgd classifier
     sgd regressor
    """
    if validation_fraction is not None and not early_stopping:
        raise ValueError("Parameter 'validation_fraction' can only be used if 'early_stopping' is set to 'True'.")

    if epsilon is not None and loss not in ["huber", "epsilon_insensitive", "squared_epsilon_insensitive"]:
        raise ValueError("Parameter 'epsilon' can only be set if "
                         "'loss' is 'huber', 'epsilon_insensitive' or 'squared_epsilon_insensitive'.")

    hp_space = dict(
        loss=loss,
        penalty=penalty or _stochastic_gradient_penalty(name_func("penalty")),
        alpha=alpha or _stochastic_gradient_alpha(name_func("alpha")),
        l1_ratio=_stochastic_gradient_l1_ratio(name_func("l1_ratio")) if l1_ratio is None else l1_ratio,
        fit_intercept=fit_intercept,
        max_iter=max_iter or _stochastic_gradient_max_iter(name_func("max_iter")),
        tol=_stochastic_gradient_tol(name_func("tol")) if tol is None else tol,
        shuffle=shuffle,
        verbose=verbose,
        epsilon=epsilon,
        random_state=_stochastic_gradient_random_state(name_func("random_state"))
        if random_state is None else random_state,
        learning_rate=learning_rate or _stochastic_gradient_learning_rate(name_func("learning_rate")),
        eta0=_stochastic_gradient_eta0(name_func("eta0")) if eta0 is None else eta0,
        power_t=_stochastic_gradient_power_t(name_func("power_t")) if power_t is None else power_t,
        early_stopping=early_stopping,
        validation_fraction=0.1 if validation_fraction is None else validation_fraction,
        n_iter_no_change=n_iter_no_change or _stochastic_gradient_n_iter_no_change(name_func("n_iter_no_change")),
        warm_start=warm_start,
        average=average,
    )
    return hp_space


@validate(params=["loss"],
          validation_test=lambda param: isinstance(param, str) and
                                        param in ["hinge", "log", "modified_huber", "squared_hinge", "perceptron",
                                                  "squared_error", "huber", "epsilon_insensitive",
                                                  "squared_epsilon_insensitive"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['hinge', 'log', 'modified_huber', "
              "'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', "
              "'squared_epsilon_insensitive'].")
@validate(params=["class_weight"],
          validation_test=lambda param: isinstance(param, str) and param == "balanced",
          msg="Invalid parameter '%s' with value '%s'. Value must be 'balanced'.")
def sgd_classifier(name: str,
                   loss: str = None,
                   n_jobs: int = 1,
                   class_weight: typing.Union[dict, str] = None,
                   **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.SGDClassifier model.

    Args:
        name: name | str
        loss: loss function to use | str
        n_jobs: number of CPUs to use | int
        class_weight: class_weight fit parameters | dict or str

    See help(hpsklearn.components.linear_model._stochastic_gradient._stochastic_gradient_hp_space)
    for info on additional available stochastic gradient arguments.
    """
    def _name(msg):
        return f"{name}.sgd_classifier_{msg}"

    sgd_loss = loss or _stochastic_gradient_classifier_loss(_name("loss"))
    hp_space = _stochastic_gradient_hp_space(_name, sgd_loss, **kwargs)
    hp_space["n_jobs"] = n_jobs
    hp_space["class_weight"] = class_weight

    return scope.sklearn_SGDClassifier(**hp_space)


@validate(params=["loss"],
          validation_test=lambda param: isinstance(param, str) and
                                        param in ["squared_error", "huber", "epsilon_insensitive",
                                                  "squared_epsilon_insensitive"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['squared_error', 'huber', "
              "'epsilon_insensitive', 'squared_epsilon_insensitive'].")
def sgd_regressor(name: str, loss: str = None, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.SGDRegressor model.

    Args:
        name: name | str
        loss: loss function to use | str

    See help(hpsklearn.components.linear_model._stochastic_gradient._stochastic_gradient_hp_space)
    for info on additional available stochastic gradient arguments.
    """
    def _name(msg):
        return f"{name}.sgd_regressor_{msg}"

    sgd_loss = loss or _stochastic_gradient_regressor_loss(_name("loss"))
    hp_space = _stochastic_gradient_hp_space(_name, sgd_loss, **kwargs)

    return scope.sklearn_SGDRegressor(**hp_space)


def sgd_one_class_svm(name: str,
                      nu: float = None,
                      fit_intercept: bool = True,
                      max_iter: int = None,
                      tol: float = None,
                      shuffle: bool = True,
                      verbose: int = 0,
                      random_state=None,
                      learning_rate: str = None,
                      eta0: float = None,
                      power_t: float = None,
                      warm_start: bool = False,
                      average: typing.Union[bool, int] = False
                      ):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.SGDOneClassSVM model.

    Args:
        name: name | str
        nu: nu parameter | float
        fit_intercept: whether to estimate intercept | bool
        max_iter: maximum number of passes | int
        tol: threshold stopping criterion | float
        shuffle: whether to shuffle training data each epoch | bool
        verbose: verbosity level | int
        random_state: seed random number generator
        learning_rate: learning rate schedule | str
        eta0: initial learning rate | float
        power_t: exponent inverse scaling lr | float
        warm_start: whether to reuse solution of previous call | bool
        average: compute average SGD weights | bool or int

    See help(hpsklearn.components.linear_model._stochastic_gradient._stochastic_gradient_hp_space)
    for info on additional available stochastic gradient arguments.
    """
    def _name(msg):
        return f"{name}.sgd_one_class_svm_{msg}"

    hp_space = dict(
        nu=_stochastic_gradient_nu(_name("nu")) if nu is None else nu,
        fit_intercept=fit_intercept,
        max_iter=max_iter or _stochastic_gradient_max_iter(_name("max_iter")),
        tol=_stochastic_gradient_tol(_name("tol")) if tol is None else tol,
        shuffle=shuffle,
        verbose=verbose,
        random_state=_stochastic_gradient_random_state(_name("random_state"))
        if random_state is None else random_state,
        learning_rate=learning_rate or _stochastic_gradient_learning_rate(_name("learning_rate")),
        eta0=_stochastic_gradient_eta0(_name("eta0")) if eta0 is None else eta0,
        power_t=_stochastic_gradient_power_t(_name("power_t")) if power_t is None else power_t,
        warm_start=warm_start,
        average=average
    )
    return scope.sklearn_SGDOneClassSVM(**hp_space)

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import naive_bayes

import numpy.typing as npt
import typing


@scope.define
def sklearn_BernoulliNB(*args, **kwargs):
    return naive_bayes.BernoulliNB(*args, **kwargs)


@scope.define
def sklearn_CategoricalNB(*args, **kwargs):
    return naive_bayes.CategoricalNB(*args, **kwargs)


@scope.define
def sklearn_ComplementNB(*args, **kwargs):
    return naive_bayes.ComplementNB(*args, **kwargs)


@scope.define
def sklearn_GaussianNB(*args, **kwargs):
    return naive_bayes.GaussianNB(*args, **kwargs)


@scope.define
def sklearn_MultinomialNB(*args, **kwargs):
    return naive_bayes.MultinomialNB(*args, **kwargs)


def _nb_hp_space(
        name_func,
        alpha: typing.Union[float, Apply] = None,
        fit_prior: typing.Union[bool, Apply] = None,
        class_prior: npt.ArrayLike = None,
        **kwargs
):
    """
    Hyper parameter search space for
     bernoulli nb
     categorical nb
     complement nb
     multinomial nb
    """
    hp_space = dict(
        alpha=hp.quniform(name_func("alpha"), 0, 1, 0.001) if alpha is None else alpha,
        fit_prior=hp.choice(name_func("fit_prior"), [True, False]) if fit_prior is None else fit_prior,
        class_prior=class_prior,
        **kwargs
    )
    return hp_space


def bernoulli_nb(name: str, binarize: typing.Union[float, None] = 0.0, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.naive_bayes.BernoulliNB model.

    Args:
        name: name | str
        binarize: threshold for binarizing | float, None

    See help(hpsklearn.components.naive_bayes._nb_hp_space)
    for info on additional available naive bayes arguments.
    """

    def _name(msg):
        return f"{name}.bernoulli_nb_{msg}"

    hp_space = _nb_hp_space(_name, **kwargs)
    hp_space["binarize"] = binarize

    return scope.sklearn_BernoulliNB(**hp_space)


def categorical_nb(name: str, min_categories: typing.Union[int, npt.ArrayLike] = None, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.naive_bayes.CategoricalNB model.

    Args:
        name: name | str
        min_categories: minimum number of categories per feature | int, ArrayLike

    See help(hpsklearn.components.naive_bayes._nb_hp_space)
    for info on additional available naive bayes arguments.
    """

    def _name(msg):
        return f"{name}.categorical_nb_{msg}"

    hp_space = _nb_hp_space(_name, **kwargs)
    hp_space["min_categories"] = min_categories

    return scope.sklearn_CategoricalNB(**hp_space)


def complement_nb(name: str, norm: typing.Union[bool, Apply] = None, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.naive_bayes.ComplementNB model.

    Args:
        name: name | str
        norm: whether a second normalization of weights is performed | bool

    See help(hpsklearn.components.naive_bayes._nb_hp_space)
    for info on additional available naive bayes arguments.
    """

    def _name(msg):
        return f"{name}.complement_nb_{msg}"

    hp_space = _nb_hp_space(_name, **kwargs)
    hp_space["norm"] = hp.choice(_name("norm"), [True, False]) if norm is None else norm

    return scope.sklearn_ComplementNB(**hp_space)


def gaussian_nb(name: str, var_smoothing: float = 1e-9, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.naive_bayes.GaussianNB model.

    Args:
        name: name | str
        var_smoothing: portion of largest variance | float
    """

    def _name(msg):
        return f"{name}.gaussian_nb_{msg}"

    hp_space = dict(
        var_smoothing=var_smoothing,
        **kwargs
    )
    return scope.sklearn_GaussianNB(**hp_space)


def multinomial_nb(name: str, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.naive_bayes.MultinomialNB model.

    Args:
        name: name | str

    See help(hpsklearn.components.naive_bayes._nb_hp_space)
    for info on additional available naive bayes arguments.
    """

    def _name(msg):
        return f"{name}.multinomial_nb_{msg}"

    hp_space = _nb_hp_space(_name, **kwargs)

    return scope.sklearn_MultinomialNB(**hp_space)

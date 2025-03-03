from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import tree
import numpy as np
import typing


@scope.define
def sklearn_DecisionTreeClassifier(*args, **kwargs):
    return tree.DecisionTreeClassifier(*args, **kwargs)


@scope.define
def sklearn_DecisionTreeRegressor(*args, **kwargs):
    return tree.DecisionTreeRegressor(*args, **kwargs)


@scope.define
def sklearn_ExtraTreeClassifier(*args, **kwargs):
    return tree.ExtraTreeClassifier(*args, **kwargs)


@scope.define
def sklearn_ExtraTreeRegressor(*args, **kwargs):
    return tree.ExtraTreeRegressor(*args, **kwargs)


def _tree_splitter(name):
    """
    Declaration search space 'splitter' parameter
    """
    return hp.choice(name, ["best", "random"])


def _tree_max_depth(name: str):
    """
    Declaration search space 'max_depth' parameter
    """
    return hp.pchoice(name, [
        (0.7, None),  # most common choice.
        (0.1, 2),  # try some shallow trees.
        (0.1, 3),
        (0.1, 4),
    ])


def _tree_min_samples_split(name: str):
    """
    Declaration search space 'min_samples_split' parameter
    """
    return hp.pchoice(name, [
        (0.95, 2),  # most common choice
        (0.05, 3),  # try minimal increase
    ])


def _tree_min_samples_leaf(name: str):
    """
    Declaration search space 'min_samples_leaf' parameter
    """
    return hp.choice(name, [
        1,  # most common choice.
        scope.int(hp.qloguniform(name + ".gt1", np.log(1.5), np.log(50.5), 1))
    ])


def _tree_min_weight_fraction_leaf(name: str):
    """
    Declaration search space 'min_weight_fraction_leaf' parameter
    """
    return hp.pchoice(name, [
        (0.95, 0.0),  # most common choice
        (0.05, 0.01),  # try minimal increase
    ])


def _tree_max_features(name: str):
    """
    Declaration search space 'max_features' parameter
    """
    return hp.pchoice(name, [
        (0.2, "sqrt"),  # most common choice.
        (0.1, "log2"),  # less common choice.
        (0.1, None),  # all features, less common choice.
        (0.6, hp.uniform(name + ".frac", 0., 1.))
    ])


def _tree_random_state(name: str):
    """
    Declaration search space 'random_state' parameter
    """
    return hp.randint(name, 5)


def _tree_min_impurity_decrease(name: str):
    """
    Declaration search space 'min_impurity_decrease' parameter
    """
    return hp.pchoice(name, [
        (0.85, 0.0),  # most common choice
        (0.05, 0.01),
        (0.05, 0.02),
        (0.05, 0.05),
    ])


def _tree_max_leaf_nodes(name: str):
    """
    Declaration search space 'max_leaf_nodes' parameter
    """
    return hp.pchoice(name, [
        (0.85, None),  # most common choice
        (0.05, 5),
        (0.05, 10),
        (0.05, 15),
    ])


@validate(params=["splitter"],
          validation_test=lambda param: not isinstance(param, str) or param in ["best", "random"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['best', 'random'].")
@validate(params=["max_features"],
          validation_test=lambda param: not isinstance(param, str) or param in ["auto", "sqrt", "log2"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['auto', 'sqrt', 'log2'].")
@validate(params=["max_depth", "min_samples_split", "min_samples_leaf", "max_features", "max_leaf_nodes",
                  "min_impurity_decrease"],
          validation_test=lambda param: not isinstance(param, float) or not param > 0,
          msg="Invalid parameter '%s' with value '%s'. Parameter value must be non-negative and greater than 0.")
@validate(params=["ccp_alpha"],
          validation_test=lambda param: not isinstance(param, float) or not param < 0,
          msg="Invalid parameter '%s' with value '%s'. Parameter value must be non-negative.")
def _tree_hp_space(
        name_func,
        splitter: typing.Union[str, Apply] = None,
        max_depth: typing.Union[int, Apply] = "Undefined",
        min_samples_split: typing.Union[float, Apply] = 2,
        min_samples_leaf: typing.Union[float, Apply] = 1,
        min_weight_fraction_leaf: typing.Union[float, Apply] = 0.0,
        max_features: typing.Union[float, str, Apply] = "Undefined",
        random_state=None,
        min_impurity_decrease: typing.Union[float, Apply] = 0.0,
        max_leaf_nodes: typing.Union[int, Apply] = "Undefined",
        ccp_alpha: float = 0.0,
        **kwargs
):
    """
    Hyper parameter search space for
     decision tree classifier
     decision tree regressor
     extra tree classifier
     extra tree regressor
    """
    hp_space = dict(
        splitter=_tree_splitter(name_func("splitter")) if splitter is None else splitter,
        max_depth=_tree_max_depth(name_func("max_depth")) if max_depth == "Undefined" else max_depth,
        min_samples_split=_tree_min_samples_split(name_func("min_samples_split"))
        if min_samples_split is None else min_samples_split,
        min_samples_leaf=_tree_min_samples_leaf(name_func("min_samples_leaf"))
        if min_samples_leaf is None else min_samples_leaf,
        min_weight_fraction_leaf=_tree_min_weight_fraction_leaf(name_func("min_weight_fraction_leaf"))
        if min_weight_fraction_leaf is None else min_weight_fraction_leaf,
        max_features=_tree_max_features(name_func("max_features")) if max_features == "Undefined" else max_features,
        random_state=_tree_random_state(name_func("random_state")) if random_state is None else random_state,
        min_impurity_decrease=_tree_min_impurity_decrease(name_func("min_impurity_decrease"))
        if min_impurity_decrease is None else min_impurity_decrease,
        max_leaf_nodes=_tree_max_leaf_nodes(name_func("max_leaf_nodes"))
        if max_leaf_nodes == "Undefined" else max_leaf_nodes,
        ccp_alpha=ccp_alpha,
        **kwargs
    )
    return hp_space


@validate(params=["criterion"],
          validation_test=lambda param: not isinstance(param, str) or param in ["gini", "entropy"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['gini', 'entropy'].")
def decision_tree_classifier(name: str,
                             criterion: typing.Union[str, Apply] = None,
                             class_weight: typing.Union[str, dict] = None,
                             **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.tree.DecisionTreeClassifier model.

    Args:
        name: name | str
        criterion: choose 'gini' or 'entropy' | str
        class_weight: weights associated with class | dict, str

    See help(hpsklearn.components.tree._classes._tree_hp_space)
    for info on additional available decision tree arguments.
    """

    def _name(msg):
        return f"{name}.dtc_{msg}"

    hp_space = _tree_hp_space(_name, **kwargs)
    hp_space["criterion"] = hp.choice(_name("criterion"), ["gini", "entropy"]) if criterion is None else criterion
    hp_space["class_weight"] = class_weight

    return scope.sklearn_DecisionTreeClassifier(**hp_space)


@validate(params=["criterion"],
          validation_test=lambda param: not isinstance(param, str) or param in ["squared_error", "friedman_mse",
                                                                                "absolute_error", "poisson"],
          msg="Invalid parameter '%s' with value '%s'. "
              "Value must be in ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'].")
def decision_tree_regressor(name: str,
                            criterion: typing.Union[str, Apply] = None,
                            **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.tree.DecisionTreeRegressor model.

    Args:
        name: name | str
        criterion: quality of split measure | str

    See help(hpsklearn.components.tree._classes._tree_hp_space)
    for info on additional available decision tree arguments.
    """

    def _name(msg):
        return f"{name}.dtr_{msg}"

    hp_space = _tree_hp_space(_name, **kwargs)
    hp_space["criterion"] = hp.choice(_name("criterion"), ["squared_error", "friedman_mse", "absolute_error"]) \
        if criterion is None else criterion

    return scope.sklearn_DecisionTreeRegressor(**hp_space)


@validate(params=["criterion"],
          validation_test=lambda param: not isinstance(param, str) or param in ["gini", "entropy"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['gini', 'entropy'].")
def extra_tree_classifier(name: str,
                          criterion: typing.Union[str, Apply] = None,
                          class_weight: typing.Union[str, dict] = None,
                          **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.tree.ExtraTreeClassifier model.

    Args:
        name: name | str
        criterion: choose 'gini' or 'entropy' | str
        class_weight: weights associated with class | dict, str

    See help(hpsklearn.components.tree._classes._tree_hp_space)
    for info on additional available extra tree arguments.
    """

    def _name(msg):
        return f"{name}.etc_{msg}"

    hp_space = _tree_hp_space(_name, **kwargs)
    hp_space["criterion"] = hp.choice(_name("criterion"), ["gini", "entropy"]) if criterion is None else criterion
    hp_space["class_weight"] = class_weight

    return scope.sklearn_ExtraTreeClassifier(**hp_space)


@validate(params=["criterion"],
          validation_test=lambda param: not isinstance(param, str) or param in ["squared_error", "friedman_mse"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['squared_error', 'friedman_mse'].")
def extra_tree_regressor(name: str,
                         criterion: typing.Union[str, Apply] = None,
                         **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.tree.ExtraTreeRegressor model.

    Args:
        name: name | str
        criterion: choose 'squared_error' or 'friedman_mse' | str

    See help(hpsklearn.components.tree._classes._tree_hp_space)
    for info on additional available extra tree arguments.
    """

    def _name(msg):
        return f"{name}.etr_{msg}"

    hp_space = _tree_hp_space(_name, **kwargs)
    hp_space["criterion"] = hp.choice(_name("criterion"), ["squared_error", "friedman_mse"]) \
        if criterion is None else criterion

    return scope.sklearn_ExtraTreeRegressor(**hp_space)

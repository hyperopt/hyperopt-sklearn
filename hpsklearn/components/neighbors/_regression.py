from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import neighbors
import typing


@scope.define
def sklearn_KNeighborsRegressor(*args, **kwargs):
    return neighbors.KNeighborsRegressor(*args, **kwargs)


@scope.define
def sklearn_RadiusNeighborsRegressor(*args, **kwargs):
    return neighbors.RadiusNeighborsRegressor(*args, **kwargs)


def _neighbors_weights(name: str):
    """
    Declaration search space 'weights' parameter
    """
    return hp.choice(name, ["uniform", "distance"])


def _neighbors_algorithm(name: str):
    """
    Declaration search space 'algorithm' parameter
    """
    return hp.choice(name, ["auto", "ball_tree", "kd_tree", "brute"])


def _neighbors_leaf_size(name: str):
    """
    Declaration search space 'leaf_size' parameter
    """
    return scope.int(hp.uniform(name, 20, 40))


def _neighbors_p(name: str):
    """
    Declaration search space 'p' parameter
    """
    return hp.uniform(name, 1, 5)


def _neighbors_metric(name: str):
    """
    Declaration search space 'metric' parameter
    """
    return hp.choice(name, ["cityblock", "l1", "l2", "minkowski", "euclidean", "manhattan"])


@validate(params=["weights"],
          validation_test=lambda param: not isinstance(param, str) or param in ["uniform", "distance"],
          msg="Invalid parameter '%s' with value '%s'. Value must be 'uniform' or 'distance'.")
@validate(params=["algorithm"],
          validation_test=lambda param: not isinstance(param, str) or param in ["auto", "ball_tree", "kd_tree",
                                                                                "brute"],
          msg="Invalid parameter '%s' with value '%s'. Value must be 'auto', 'ball_tree', 'kd_tree', or 'brute'.")
def neighbors_hp_space(
        name_func,
        weights: typing.Union[str, callable, Apply] = None,
        algorithm: typing.Union[str, Apply] = None,
        leaf_size: typing.Union[int, Apply] = None,
        p: typing.Union[int, Apply] = None,
        metric: typing.Union[str, callable, Apply] = None,
        metric_params: dict = None,
        n_jobs: int = 1,
        **kwargs):
    """
    Hyper parameter search space for
     k neighbors regressor
     radius neighbors regressor
    """
    hp_space = dict(
        weights=_neighbors_weights(name_func("weights")) if weights is None else weights,
        algorithm=_neighbors_algorithm(name_func("algorithm")) if algorithm is None else algorithm,
        leaf_size=_neighbors_leaf_size(name_func("leaf_size")) if leaf_size is None else leaf_size,
        p=_neighbors_p(name_func("p")) if p is None else p,
        metric=_neighbors_metric(name_func("metric")) if metric is None else metric,
        metric_params=metric_params,
        n_jobs=n_jobs,
        **kwargs
    )
    return hp_space


def k_neighbors_regressor(name: str,
                          n_neighbors: typing.Union[int, Apply] = None,
                          **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.neighbors.KNeighborsRegressor model.

    Args:
        name: name | str
        n_neighbors: number of neighbors | int

    See help(hpsklearn.components.neighbors._regression._neighbors_hp_space)
    for info on additional available neighbors regression arguments.
    """

    def _name(msg):
        return f"{name}.k_neighbors_regressor_{msg}"

    hp_space = neighbors_hp_space(_name, **kwargs)
    hp_space["n_neighbors"] = scope.int(hp.uniform(_name("n_neighbors"), 1, 15)) if n_neighbors is None else n_neighbors

    return scope.sklearn_KNeighborsRegressor(**hp_space)


def radius_neighbors_regressor(name: str,
                               radius: typing.Union[float, Apply] = None,
                               **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.neighbors.RadiusNeighborsRegressor model.

    Args:
        name: name | str
        radius: range of parameter space | float

    See help(hpsklearn.components.neighbors._regression._neighbors_hp_space)
    for info on additional available neighbors arguments.
    """

    def _name(msg):
        return f"{name}.radius_neighbors_regressor_{msg}"

    hp_space = neighbors_hp_space(_name, **kwargs)
    hp_space["radius"] = hp.uniform(_name("radius"), 25, 100) if radius is None else radius  # very dependent on data

    return scope.sklearn_RadiusNeighborsRegressor(**hp_space)

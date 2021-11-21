from hpsklearn.components._base import validate

from hyperopt.pyll import scope
from hyperopt import hp

from ._regression import neighbors_hp_space
from sklearn import neighbors
import typing


@scope.define
def sklearn_KNeighborsClassifier(*args, **kwargs):
    return neighbors.KNeighborsClassifier(*args, **kwargs)


@scope.define
def sklearn_RadiusNeighborsClassifier(*args, **kwargs):
    return neighbors.RadiusNeighborsClassifier(*args, **kwargs)


def k_neighbors_classifier(name: str,
                           n_neighbors: int = None,
                           **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.neighbors.KNeighborsClassifier model.

    Args:
        name: name | str
        n_neighbors: number of neighbors | int

    See help(hpsklearn.components.neighbors._regression._neighbors_regression_hp_space)
    for info on additional available neighbors arguments.
    """
    def _name(msg):
        return f"{name}.k_neighbors_classifier_{msg}"

    hp_space = neighbors_hp_space(_name, **kwargs)
    hp_space["n_neighbors"] = scope.int(hp.uniform(_name("n_neighbors"), 1, 15)) if n_neighbors is None else n_neighbors

    return scope.sklearn_KNeighborsClassifier(**hp_space)


@validate(params=["outlier_label"],
          validation_test=lambda param: param == "most_frequent",
          msg="Invalid parameter '%s' with value '%s'. Value must be 'most_frequent'.")
def radius_neighbors_classifier(name: str,
                                radius: float = None,
                                outlier_label: typing.Union[int, str] = None,
                                **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.neighbors.RadiusNeighborsClassifier model.

    Args:
        name: name | str
        radius: range of parameter space | float
        outlier_label: label for outlier samples | int | str

    See help(hpsklearn.components.neighbors._regression._neighbors_regression_hp_space)
    for info on additional available neighbors arguments.
    """
    def _name(msg):
        return f"{name}.radius_neighbors_classifier_{msg}"

    hp_space = neighbors_hp_space(_name, **kwargs)
    hp_space["radius"] = hp.uniform(_name("radius"), 0.5, 10) if radius is None else radius
    hp_space["outlier_label"] = outlier_label

    return scope.sklearn_RadiusNeighborsClassifier(**hp_space)
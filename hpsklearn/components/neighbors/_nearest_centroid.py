from hyperopt.pyll import scope
from hyperopt import hp

from sklearn import neighbors
import typing


@scope.define
def sklearn_NearestCentroid(*args, **kwargs):
    return neighbors.NearestCentroid(*args, **kwargs)


def nearest_centroid(name: str,
                     metric: typing.Union[str, callable] = None,
                     shrink_threshold: float = None):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.neighbors.NearestCentroid model.

    Args:
        name: name | str
        metric: metric to use | str, callable
        shrink_threshold: shrink threshold | float
    """
    def _name(msg):
        return f"{name}.nearest_centroid_{msg}"

    hp_space = dict(
        metric=metric or hp.choice(_name("metric"), ["cityblock", "cosine", "l1", "l2",
                                                     "minkowski", "euclidean", "manhattan"]),
        shrink_threshold=shrink_threshold,
    )
    return scope.sklearn_NearestCentroid(**hp_space)

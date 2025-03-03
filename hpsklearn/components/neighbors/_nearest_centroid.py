from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import neighbors
import typing


@scope.define
def sklearn_NearestCentroid(*args, **kwargs):
    return neighbors.NearestCentroid(*args, **kwargs)


def nearest_centroid(name: str,
                     metric: typing.Union[str, Apply] = None,
                     shrink_threshold: float = None,
                     **kwargs):
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
        metric=hp.choice(_name("metric"), ["euclidean", "manhattan"])
        if metric is None else metric,
        shrink_threshold=shrink_threshold,
        **kwargs,
    )
    return scope.sklearn_NearestCentroid(**hp_space)

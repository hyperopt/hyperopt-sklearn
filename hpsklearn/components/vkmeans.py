from hpsklearn.components._base import validate
from hpsklearn.objects import ColumnKMeans

from hyperopt.pyll import scope, Apply
from hyperopt import hp

import numpy.typing as npt
import numpy as np
import typing


@scope.define
def sklearn_ColumnKMeans(*args, **kwargs):
    return ColumnKMeans(*args, **kwargs)


@validate(params=["max_iter", "n_clusters"],
          validation_test=lambda param: param > 0,
          msg="Invalid parameter '%s' with value '%s'. Value must be 0 or higher.")
def colkmeans(name: str,
              n_clusters: typing.Union[int, Apply] = None,
              init: typing.Union[callable, npt.ArrayLike, str, Apply] = None,
              n_init: typing.Union[int, Apply] = None,
              max_iter: typing.Union[int, Apply] = None,
              tol: typing.Union[float, Apply] = None,
              verbose: int = 0,
              random_state=None,
              copy_x: bool = True):
    """
    Return a pyll graph with hyperparameters that will construct
    a hpsklearn.objects.ColumnKMeans transformer.

    Args:
        name: name | str
        n_clusters: number of clusters | int
        init: initialization method for the centroids | callable, array-like, str
        n_init: number of times to run k-means algorithm | int
        max_iter: maximum number of iterations | int
        tol: relative tolerance in regard to Frobenius norm | float
        precompute_distances: precompute distances | bool
        verbose: verbosity level | int
        random_state: random seed | int
        copy_x: modify copy of data | bool
        n_jobs: number of jobs to run in parallel | int
    """

    rval = scope.sklearn_ColumnKMeans(
        n_clusters=scope.int(hp.qloguniform(name + ".n_clusters", low=np.log(1.51), high=np.log(19.5), q=1.0))
        if n_clusters is None else n_clusters,
        init=hp.choice(name + ".init", ["k-means++", "random"]) if init is None else init,
        n_init=hp.choice(name + ".n_init", [1, 2, 10, 20]) if n_init is None else n_init,
        max_iter=scope.int(hp.qlognormal(name + ".max_iter", np.log(300), np.log(10), q=1))
        if max_iter is None else max_iter,
        tol=hp.lognormal(name + ".tol", np.log(0.0001), np.log(10)) if tol is None else tol,
        verbose=verbose,
        random_state=random_state,
        copy_x=copy_x,
    )
    return rval

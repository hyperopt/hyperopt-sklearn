from hpsklearn.components._base import validate
from hpsklearn.objects import ColumnKMeans

from hyperopt.pyll import scope
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
              n_clusters: int = None,
              init: typing.Union[callable, npt.ArrayLike, str] = None,
              n_init: int = None,
              max_iter: int = None,
              tol: float = None,
              precompute_distances: bool = True,
              verbose: int = 0,
              random_state=None,
              copy_x: bool = True,
              n_jobs: int = 1):
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
        n_clusters=n_clusters or
        scope.int(hp.qloguniform(name + ".n_clusters", low=np.log(1.51), high=np.log(19.5), q=1.0)),
        init=init or hp.choice(name + ".init", ["k-means++", "random"]),
        n_init=hp.choice(name + ".n_init", [1, 2, 10, 20]) if n_init is None else n_init,
        max_iter=max_iter or scope.int(hp.qlognormal(name + ".max_iter", np.log(300), np.log(10), q=1)),
        tol=hp.lognormal(name + ".tol", np.log(0.0001), np.log(10)) if tol is None else tol,
        precompute_distances=precompute_distances,
        verbose=verbose,
        random_state=random_state,
        copy_x=copy_x,
        n_jobs=n_jobs,
    )
    return rval

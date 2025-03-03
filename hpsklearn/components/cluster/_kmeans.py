from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import cluster
import numpy.typing as npt
import typing


@scope.define
def sklearn_KMeans(*args, **kwargs):
    return cluster.KMeans(*args, **kwargs)


@scope.define
def sklearn_MiniBatchKMeans(*args, **kwargs):
    return cluster.MiniBatchKMeans(*args, **kwargs)


def _kmeans_n_clusters(name: str):
    """
    Declaration search space 'n_clusters' parameter
    """
    return scope.int(hp.uniform(name, 1, 20))


def _kmeans_init(name: str):
    """
    Declaration search space 'init' parameter
    """
    return hp.choice(name, ["k-means++", "random"])


def _kmeans_random_state(name: str):
    """
    Declaration search space 'random_state' parameter
    """
    return hp.randint(name, 5)


def _kmeans_hp_space(
        name_func,
        n_clusters: typing.Union[int, Apply] = None,
        init: typing.Union[str, callable, npt.ArrayLike, Apply] = None,
        verbose: int = 0,
        random_state=None,
        **kwargs
):
    """
    Hyper parameter search space for
     k means
     mini batch k means
    """
    hp_space = dict(
        n_clusters=_kmeans_n_clusters(name_func("n_clusters")) if n_clusters is None else n_clusters,
        init=_kmeans_init(name_func("init")) if init is None else init,
        verbose=verbose,
        random_state=_kmeans_random_state(name_func("random_state")) if random_state is None else random_state,
        **kwargs
    )
    return hp_space


@validate(params=["algorithm"],
          validation_test=lambda param: not isinstance(param, str) or param in ["lloyd", "elkan"],
          msg="Invalid parameter '%s' with value '%s'. Value must be 'lloyd' or 'elkan'")
def k_means(name: str,
            n_init: typing.Union[int, Apply] = None,
            max_iter: typing.Union[int, Apply] = None,
            tol: typing.Union[float, Apply] = None,
            copy_x: bool = True,
            algorithm: typing.Union[str, Apply] = None,
            **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.cluster.KMeans model.

    Args:
        name: name | str
        n_init: number of times to run k-means algorithm | int
        max_iter: maximum number of iterations | int
        tol: relative tolerance in regard to Frobenius norm | float
        copy_x: modify copy of data | bool
        algorithm: K-means algorithm to use | str

    See help(hpsklearn.components.cluster._kmeans._kmeans_hp_space)
    for info on additional available k means arguments.
    """

    def _name(msg):
        return f"{name}.k_means_{msg}"

    hp_space = _kmeans_hp_space(_name, **kwargs)
    hp_space["n_init"] = scope.int(hp.uniform(_name("n_init"), 2, 25)) if n_init is None else n_init
    hp_space["max_iter"] = scope.int(hp.uniform(_name("max_iter"), 100, 500)) if max_iter is None else max_iter
    hp_space["tol"] = hp.uniform(_name("tol"), 1e-5, 1e-3) if tol is None else tol
    hp_space["copy_x"] = copy_x
    hp_space["algorithm"] = hp.choice(_name("algorithm"), ["lloyd", "elkan"]) if algorithm is None else algorithm

    return scope.sklearn_KMeans(**hp_space)


def mini_batch_k_means(name: str,
                       max_iter: typing.Union[int, Apply] = None,
                       batch_size: typing.Union[int, Apply] = None,
                       compute_labels: bool = True,
                       tol: typing.Union[float, Apply] = None,
                       max_no_improvement: typing.Union[int, Apply] = None,
                       init_size: int = None,
                       n_init: typing.Union[int, Apply] = None,
                       reassignment_ratio: typing.Union[float, Apply] = None,
                       **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.cluster.KMeans model.

    Args:
        name: name | str
        max_iter: maximum number of iterations | int
        batch_size: size of the mini batches | int
        compute_labels: compute label assignment and inertia | bool
        tol: relative tolerance with regards to Frobenius norm | float
        max_no_improvement: early stopping when no improvement found | int
        init_size: random samples for initialization | int
        n_init: number of times to run k-means algorithm | int
        reassignment_ratio: control the fraction for center reassignment | float

    See help(hpsklearn.components.cluster._kmeans._kmeans_hp_space)
    for info on additional available k means arguments.
    """

    def _name(msg):
        return f"{name}.mini_batch_k_means_{msg}"

    hp_space = _kmeans_hp_space(_name, **kwargs)
    hp_space["max_iter"] = scope.int(hp.uniform(_name("max_iter"), 100, 300)) if max_iter is None else max_iter
    hp_space["batch_size"] = hp.choice(_name("batch_size"), [256, 512, 1024, 2048]) \
        if batch_size is None else batch_size
    hp_space["compute_labels"] = compute_labels
    hp_space["tol"] = hp.uniform(_name("tol"), 1e-7, 1e-5) if tol is None else tol
    hp_space["max_no_improvement"] = scope.int(hp.uniform(_name("max_no_improvement"), 5, 25)) \
        if max_no_improvement is None else max_no_improvement
    hp_space["init_size"] = init_size
    hp_space["n_init"] = hp.choice(_name("n_init"), [1, 2, 3, 4]) if n_init is None else n_init
    hp_space["reassignment_ratio"] = hp.uniform(_name("reassignment_ratio"), 0.001, 0.1) \
        if reassignment_ratio is None else reassignment_ratio

    return scope.sklearn_MiniBatchKMeans(**hp_space)

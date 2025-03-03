from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import multiclass
from . import any_classifier
import typing


@scope.define
def sklearn_OneVsRestClassifier(*args, **kwargs):
    return multiclass.OneVsRestClassifier(*args, **kwargs)


@scope.define
def sklearn_OneVsOneClassifier(*args, **kwargs):
    return multiclass.OneVsOneClassifier(*args, **kwargs)


@scope.define
def sklearn_OutputCodeClassifier(*args, **kwargs):
    return multiclass.OutputCodeClassifier(*args, **kwargs)


def one_vs_rest_classifier(name: str,
                           estimator: typing.Union[object, Apply] = None,
                           n_jobs: int = 1,
                           **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.multiclass.OneVsRestClassifier model.

    Args:
        name: name | str
        estimator: estimator object | object
        n_jobs: number of CPUs to use | int
    """

    def _name(msg):
        return f"{name}.one_vs_rest_{msg}"

    hp_space = dict(
        estimator=any_classifier(_name("estimator")) if estimator is None else estimator,
        n_jobs=n_jobs,
        **kwargs
    )
    return scope.sklearn_OneVsRestClassifier(**hp_space)


def one_vs_one_classifier(name: str,
                          estimator: typing.Union[object, Apply] = None,
                          n_jobs: int = 1,
                          **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.multiclass.OneVsOneClassifier model.

    Args:
        name: name | str
        estimator: estimator object | object
        n_jobs: number of CPUs to use | int
    """

    def _name(msg):
        return f"{name}.one_vs_one_{msg}"

    hp_space = dict(
        estimator=any_classifier(_name("estimator")) if estimator is None else estimator,
        n_jobs=n_jobs,
        **kwargs
    )
    return scope.sklearn_OneVsOneClassifier(**hp_space)


def output_code_classifier(name: str,
                           estimator: typing.Union[object, Apply] = None,
                           code_size: typing.Union[float, Apply] = None,
                           random_state=None,
                           n_jobs: int = 1,
                           **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.multiclass.OutputCodeClassifier model.

    Args:
        name: name | str
        estimator: estimator object | object
        code_size: percentage number of classes | float
        random_state: random state | int
        n_jobs: number of CPUs to use | int
    """

    def _name(msg):
        return f"{name}.output_code_classifier{msg}"

    hp_space = dict(
        estimator=any_classifier(_name("estimator")) if estimator is None else estimator,
        code_size=hp.uniform(_name("code_size"), 1, 2) if code_size is None else code_size,
        n_jobs=n_jobs,
        random_state=hp.randint(_name("random_state"), 5) if random_state is None else random_state,
        **kwargs
    )
    return scope.sklearn_OutputCodeClassifier(**hp_space)

from hyperopt.pyll import scope
from hyperopt import hp

from sklearn import multiclass
from . import any_classifier


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
                           estimator: object = None,
                           n_jobs: int = 1):
    def _name(msg):
        return f"{name}.one_vs_rest_{msg}"

    hp_space = dict(
        estimator=estimator or any_classifier(_name("estimator")),
        n_jobs=n_jobs
    )
    return scope.sklearn_OneVsRestClassifier(**hp_space)


def one_vs_one_classifier(name: str,
                          estimator: object = None,
                          n_jobs: int = 1):
    def _name(msg):
        return f"{name}.one_vs_one_{msg}"

    hp_space = dict(
        estimator=estimator or any_classifier(_name("estimator")),
        n_jobs=n_jobs
    )
    return scope.sklearn_OneVsOneClassifier(**hp_space)


def output_code_classifier(name: str,
                           estimator: object = None,
                           code_size: float = None,
                           random_state=None,
                           n_jobs: int = 1):
    def _name(msg):
        return f"{name}.output_code_classifier{msg}"

    hp_space = dict(
        estimator=estimator or any_classifier(_name("estimator")),
        code_size=hp.uniform(_name("code_size"), 1, 2) if code_size is None else code_size,
        n_jobs=n_jobs,
        random_state=hp.randint(_name("random_state"), 5) if random_state is None else random_state
    )
    return scope.sklearn_OutputCodeClassifier(**hp_space)

from sklearn import ensemble
from hyperopt.pyll import scope


@scope.define
def sklearn_BaggingClassifier(*args, **kwargs):
    return ensemble.BaggingClassifier(*args, **kwargs)


@scope.define
def sklearn_BaggingRegressor(*args, **kwargs):
    return ensemble.BaggingRegressor(*args, **kwargs)

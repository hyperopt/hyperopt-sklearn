from sklearn import ensemble
from hyperopt.pyll import scope


@scope.define
def sklearn_StackingClassifier(*args, **kwargs):
    return ensemble.StackingClassifier(*args, **kwargs)


@scope.define
def sklearn_StackingRegressor(*args, **kwargs):
    return ensemble.StackingRegressor(*args, **kwargs)

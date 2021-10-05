from sklearn import ensemble
from hyperopt.pyll import scope


@scope.define
def sklearn_AdaBoostClassifier(*args, **kwargs):
    return ensemble.AdaBoostClassifier(*args, **kwargs)


@scope.define
def sklearn_AdaBoostRegressor(*args, **kwargs):
    return ensemble.AdaBoostRegressor(*args, **kwargs)

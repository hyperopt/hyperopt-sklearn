from sklearn import ensemble
from hyperopt.pyll import scope


@scope.define
def sklearn_GradientBoostingClassifier(*args, **kwargs):
    return ensemble.GradientBoostingClassifier(*args, **kwargs)


@scope.define
def sklearn_GradientBoostingRegressor(*args, **kwargs):
    return ensemble.GradientBoostingRegressor(*args, **kwargs)

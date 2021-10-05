from sklearn import ensemble
from hyperopt.pyll import scope


@scope.define
def sklearn_HistGradientBoostingClassifier(*args, **kwargs):
    return ensemble.HistGradientBoostingClassifier(*args, **kwargs)


@scope.define
def sklearn_HistGradientBoostingRegressor(*args, **kwargs):
    return ensemble.HistGradientBoostingRegressor(*args, **kwargs)

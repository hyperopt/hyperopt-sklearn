from sklearn import ensemble
from hyperopt.pyll import scope


@scope.define
def sklearn_IsolationForest(*args, **kwargs):
    return ensemble.IsolationForest(*args, **kwargs)

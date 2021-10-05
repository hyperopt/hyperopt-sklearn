from sklearn import ensemble
from hyperopt.pyll import scope


@scope.define
def sklearn_VotingClassifier(*args, **kwargs):
    return ensemble.VotingClassifier(*args, **kwargs)


@scope.define
def sklearn_VotingRegressor(*args, **kwargs):
    return ensemble.VotingRegressor(*args, **kwargs)

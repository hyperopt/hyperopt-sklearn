from hpsklearn.objects import LagSelector

from hyperopt.pyll import scope
from hyperopt import hp


@scope.define
def ts_LagSelector(*args, **kwargs):
    return LagSelector(*args, **kwargs)


def ts_lagselector(name: str, lower_lags: float = 1, upper_lags: float = 1):
    """
    Return a pyll graph with hyperparameters that will construct
    a hpsklearn.objects.LagSelector transformer.

    Args:
        name: name | str
        lower_lags: lower lag size | float
        upper_lags: upper lag size | float
    """
    rval = scope.ts_LagSelector(
        lag_size=scope.int(hp.quniform(name + ".lags", lower_lags - .5, upper_lags + .5, 1))
    )
    return rval

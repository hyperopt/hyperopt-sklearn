from hpsklearn.components._base import validate

from hyperopt.pyll import scope
from hyperopt import hp

from sklearn import preprocessing
import numpy.typing as npt
import typing


@scope.define
def sklearn_KBinsDiscretizer(*args, **kwargs):
    return preprocessing.KBinsDiscretizer(*args, **kwargs)


@validate(params=["n_bins"],
          validation_test=lambda param: isinstance(param, npt.ArrayLike) or param >= 2,
          msg="Invalid parameter '%s' with value '%s'. Value must be 2 or higher.")
@validate(params=["encode"],
          validation_test=lambda param: param in ["onehot", "onehot-dense", "ordinal"],
          msg="Invalid parameter '%s' with value '%s'. Choose 'onehot', 'onehot-dense' or 'ordinal'.")
@validate(params=["strategy"],
          validation_test=lambda param: param in ["uniform", "quantile", "kmeans"],
          msg="Invalid parameter '%s' with value '%s'. Choose 'uniform', 'quantile' or 'kmeans'.")
def k_bins_discretizer(name: str,
                       n_bins: typing.Union[int, npt.ArrayLike] = None,
                       encode: str = None,
                       strategy: str = None,
                       dtype=None):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.preprocessing.KBinsDiscretizer transformer.

    Args:
        name: name | str
        n_bins: number of bins | int, npt.ArrayLike
        encode: encoding method | str
        strategy: strategy used to define width of bins | str
        dtype: dtype of output | type
    """
    rval = scope.sklearn_KBinsDiscretizer(
        n_bins=n_bins or scope.int(hp.uniform(name + ".n_bins", 2, 20)),
        encode=encode or hp.choice(name + ".encode", ["onehot-dense", "ordinal"]),
        strategy=strategy or hp.choice(name + ".strategy", ["uniform", "quantile", "kmeans"]),
        dtype=dtype
    )

    return rval

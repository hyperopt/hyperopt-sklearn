import typing

from hpsklearn.components._base import validate

from hyperopt.pyll import scope
from hyperopt import hp

from sklearn import preprocessing
import numpy as np


@scope.define
def sklearn_MinMaxScaler(*args, **kwargs):
    return preprocessing.MinMaxScaler(*args, **kwargs)


@scope.define
def sklearn_Normalizer(*args, **kwargs):
    return preprocessing.Normalizer(*args, **kwargs)


@scope.define
def sklearn_OneHotEncoder(*args, **kwargs):
    return preprocessing.OneHotEncoder(*args, **kwargs)


@scope.define
def sklearn_StandardScaler(*args, **kwargs):
    return preprocessing.StandardScaler(*args, **kwargs)


def min_max_scaler(name: str,
                   feature_range: tuple = None,
                   copy: bool = True,
                   clip: bool = None):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.preprocessing.MinMaxScaler transformer.

    Args:
        name: name | str
        feature_range: desired range of transformed data | tuple
        copy: perform inplace row normalization or on copy | bool
        clip: clip transformed values of held-out data to provided 'feature range' | bool
    """
    rval = scope.sklearn_MinMaxScaler(
        feature_range=feature_range or (hp.choice(name + ".feature_min", [-1.0, 0.0]), 1.0),
        copy=copy,
        clip=clip or hp.choice(name + ".clip", [True, False])
    )

    return rval


@validate(params=["norm"],
          validation_test=lambda param: isinstance(param, str) and param in ("l1", "l2", "max"),
          msg="Invalid parameter '%s' with value '%s'. Choose 'l1', 'l2' or 'max'.")
def normalizer(name: str,
               norm: str = None,
               copy: bool = True):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.preprocessing.Normalizer transformer.

    Args:
        name: name | str
        norm: choose 'l1', 'l2' or 'max' | str
        copy: perform inplace row normalization or on copy | bool
    """
    rval = scope.sklearn_Normalizer(
        norm=norm or hp.choice(name + ".norm", ["l1", "l2", "max"]),
        copy=copy
    )

    return rval


@validate(params=["categories"],
          validation_test=lambda param: isinstance(param, str) and param == "auto",
          msg="Invalid parameter '%s' with value '%s'. Choose 'auto' or a list of array-like.")
@validate(params=["drop"],
          validation_test=lambda param: isinstance(param, str) and param in ("first", "if_binary"),
          msg="Invalid parameter '%s' with value '%s'. Choose 'first' or 'if_binary'.")
def one_hot_encoder(name: str,
                    categories: typing.Union[str, list] = "auto",
                    drop: typing.Union[str, np.ndarray] = None,
                    sparse: bool = True,
                    dtype: type = np.float64):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.preprocessing.OneHotEncoder transformer.

    Args:
        name: name | str
        categories: categories per feature | str or list
        drop: choose 'first' or 'if_binary' | str or np.ndarray
        sparse: return sparse matrix or array | bool
        dtype: desired dtype of output | type
    """
    rval = scope.sklearn_OneHotEncoder(
        categories=categories,
        drop=drop or hp.choice(name + ".drop", ["first", "if_binary"]),
        sparse=sparse,
        dtype=dtype
    )

    return rval


def standard_scaler(name: str,
                    copy: bool = True,
                    with_mean: bool = None,
                    with_std: bool = None):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.preprocessing.StandardScaler transformer.

    Args:
         name: name | str
         copy: perform inplace scaling or on copy | bool
         with_mean: center data before scaling | bool
         with_std: scale data to unit variance | bool
    """
    rval = scope.sklearn_StandardScaler(
        copy=copy,
        with_mean=with_mean or hp.choice(name + ".with_mean", [True, False]),
        with_std=with_std or hp.choice(name + ".with_std", [True, False])
    )

    return rval

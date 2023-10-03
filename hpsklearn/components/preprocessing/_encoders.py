from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import preprocessing
import numpy as np
import typing


@scope.define
def sklearn_OneHotEncoder(*args, **kwargs):
    return preprocessing.OneHotEncoder(*args, **kwargs)


@scope.define
def sklearn_OrdinalEncoder(*args, **kwargs):
    return preprocessing.OrdinalEncoder(*args, **kwargs)


@validate(params=["categories"],
          validation_test=lambda param: not isinstance(param, str) or param == "auto",
          msg="Invalid parameter '%s' with value '%s'. Choose 'auto' or a list of array-like.")
@validate(params=["drop"],
          validation_test=lambda param: not isinstance(param, str) or param in ("first", "if_binary"),
          msg="Invalid parameter '%s' with value '%s'. Choose 'first' or 'if_binary'.")
def one_hot_encoder(name: str,
                    categories: typing.Union[str, list] = "auto",
                    drop: typing.Union[str, np.ndarray, Apply] = None,
                    sparse_output: bool = True,
                    dtype: type = np.float64):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.preprocessing.OneHotEncoder transformer.

    Args:
        name: name | str
        categories: categories per feature | str or list
        drop: choose 'first' or 'if_binary' | str or np.ndarray
        sparse_output: return sparse_output matrix or array | bool
        dtype: desired dtype of output | type
    """
    rval = scope.sklearn_OneHotEncoder(
        categories=categories,
        drop=hp.choice(name + ".drop", ["first", "if_binary"]) if drop is None else drop,
        sparse_output=sparse_output,
        dtype=dtype
    )

    return rval


@validate(params=["categories"],
          validation_test=lambda param: not isinstance(param, str) or param == "auto",
          msg="Invalid parameter '%s' with value '%s'. Choose 'auto' or a list of array-like.")
@validate(params=["handle_unknown"],
          validation_test=lambda param: not isinstance(param, str) or param in ("error", "use_encoded_value"),
          msg="Invalid parameter '%s' with value '%s'. Choose 'error' or 'use_encoded_value'.")
def ordinal_encoder(name: str,
                    categories: typing.Union[str, list] = "auto",
                    dtype: type = np.float64,
                    handle_unknown: str = "error",
                    unknown_value: float = None):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.preprocessing.OrdinalEncoder transformer.

    Args:
        name: name | str
        categories: categories per feature | str or list
        dtype: desired dtype of output | type
        handle_unknown: choose 'error' or 'use_encoded_value' | str
        unknown_value: value for unknown categories | int or np.nan
    """
    rval = scope.sklearn_OrdinalEncoder(
        categories=categories,
        dtype=dtype,
        handle_unknown=handle_unknown,
        unknown_value=unknown_value
    )

    return rval

from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import preprocessing
import typing


@scope.define
def sklearn_Binarizer(*args, **kwargs):
    return preprocessing.Binarizer(*args, **kwargs)


@scope.define
def sklearn_MinMaxScaler(*args, **kwargs):
    return preprocessing.MinMaxScaler(*args, **kwargs)


@scope.define
def sklearn_MaxAbsScaler(*args, **kwargs):
    return preprocessing.MaxAbsScaler(*args, **kwargs)


@scope.define
def sklearn_Normalizer(*args, **kwargs):
    return preprocessing.Normalizer(*args, **kwargs)


@scope.define
def sklearn_RobustScaler(*args, **kwargs):
    return preprocessing.RobustScaler(*args, **kwargs)


@scope.define
def sklearn_StandardScaler(*args, **kwargs):
    return preprocessing.StandardScaler(*args, **kwargs)


@scope.define
def sklearn_QuantileTransformer(*args, **kwargs):
    return preprocessing.QuantileTransformer(*args, **kwargs)


@scope.define
def sklearn_PowerTransformer(**kwargs):
    return preprocessing.PowerTransformer(**kwargs)


def binarizer(name: str,
              threshold: typing.Union[float, Apply] = None,
              copy: bool = True):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.preprocessing.Binarizer transformer.

    Args:
        name: name | str
        threshold: threshold for binarization | float
        copy: perform inplace binarization or on copy | bool
    """
    rval = scope.sklearn_Binarizer(
        threshold=hp.uniform(name + ".threshold", 0.0, 1.0) if threshold is None else threshold,
        copy=copy
    )

    return rval


def min_max_scaler(name: str,
                   feature_range: typing.Union[tuple, Apply] = None,
                   copy: bool = True,
                   clip: typing.Union[bool, Apply] = None):
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
        feature_range=(hp.choice(name + ".feature_min", [-1.0, 0.0]), 1.0) if feature_range is None else feature_range,
        copy=copy,
        clip=hp.choice(name + ".clip", [True, False]) if clip is None else clip
    )

    return rval


def max_abs_scaler(name: str,
                   copy: bool = True):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.preprocessing.MaxAbsScaler transformer.

    Args:
        name: name | str
        copy: perform inplace row normalization or on copy | bool
    """
    rval = scope.sklearn_MaxAbsScaler(
        copy=copy
    )

    return rval


@validate(params=["norm"],
          validation_test=lambda param: not isinstance(param, str) or param in ("l1", "l2", "max"),
          msg="Invalid parameter '%s' with value '%s'. Choose 'l1', 'l2' or 'max'.")
def normalizer(name: str,
               norm: typing.Union[str, Apply] = None,
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
        norm=hp.choice(name + ".norm", ["l1", "l2", "max"]) if norm is None else norm,
        copy=copy
    )

    return rval


def robust_scaler(name: str,
                  with_centering: typing.Union[bool, Apply] = None,
                  with_scaling: typing.Union[bool, Apply] = None,
                  quantile_range: typing.Union[tuple, Apply] = None,
                  copy: bool = True,
                  unit_variance: typing.Union[bool, Apply] = None):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.preprocessing.RobustScaler transformer.

    Args:
        name: name | str
        with_centering: center the data with median absolute deviation | bool
        with_scaling: scale the data with median absolute deviation | bool
        quantile_range: range of quantiles | tuple
        copy: perform inplace row normalization or on copy | bool
        unit_variance: ensure that the variance of the data is 1 | bool
    """
    rval = scope.sklearn_RobustScaler(
        with_centering=hp.choice(name + ".with_centering", [True, False]) if with_centering is None else with_centering,
        with_scaling=hp.choice(name + ".with_scaling", [True, False]) if with_scaling is None else with_scaling,
        quantile_range=(hp.uniform(name + ".quantile_min", 0.0, 0.5), hp.uniform(name + ".quantile_max", 0.5, 1.0))
        if quantile_range is None else quantile_range,
        copy=copy,
        unit_variance=hp.choice(name + ".unit_variance", [True, False]) if unit_variance is None else unit_variance
    )

    return rval


def standard_scaler(name: str,
                    copy: bool = True,
                    with_mean: typing.Union[bool, Apply] = None,
                    with_std: typing.Union[bool, Apply] = None):
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
        with_mean=hp.choice(name + ".with_mean", [True, False]) if with_mean is None else with_mean,
        with_std=hp.choice(name + ".with_std", [True, False]) if with_std is None else with_std
    )

    return rval


@validate(params=["output_distribution"],
          validation_test=lambda param: not isinstance(param, str) or param in ("normal", "uniform"),
          msg="Invalid parameter '%s' with value '%s'. Choose 'normal' or 'uniform'.")
def quantile_transformer(name: str,
                         n_quantiles: typing.Union[int, Apply] = None,
                         output_distribution: typing.Union[str, Apply] = None,
                         subsample: typing.Union[int, Apply] = None,
                         random_state=None,
                         copy: bool = True):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.preprocessing.QuantileTransformer transformer.

    Args:
        name: name | str
        n_quantiles: number of quantiles | int
        output_distribution: choose 'uniform', 'normal' | str
        subsample: subsample size of training data | int
        random_state: random seed
        copy: perform inplace row normalization or on copy | bool
    """
    rval = scope.sklearn_QuantileTransformer(
        n_quantiles=scope.int(hp.uniform(name + ".n_quantiles", 500, 1500)) if n_quantiles is None else n_quantiles,
        output_distribution=hp.choice(name + ".output_distribution", ["normal", "uniform"])
        if output_distribution is None else output_distribution,
        subsample=scope.int(hp.uniform(name + ".subsample", 1e4, 1e6)) if subsample is None else subsample,
        random_state=hp.randint(name + ".random_state", 5) if random_state is None else random_state,
        copy=copy
    )

    return rval


def power_transformer(name: str,
                      method: typing.Union[str, Apply] = None,
                      standardize: typing.Union[bool, Apply] = None,
                      copy: bool = True):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.preprocessing.PowerTransformer transformer.

    Args:
        name: name | str
        method: method to use for transformation | str
        standardize: standardize data to zero mean and unit variance | bool
        copy: perform inplace row normalization or on copy | bool
    """
    rval = scope.sklearn_PowerTransformer(
        method=hp.choice(name + ".method", ["yeo-johnson", "box-cox"]) if method is None else method,
        standardize=hp.choice(name + ".standardize", [True, False]) if standardize is None else standardize,
        copy=copy
    )

    return rval

from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import preprocessing
import numpy.typing as npt
import typing


@scope.define
def sklearn_PolynomialFeatures(*args, **kwargs):
    return preprocessing.PolynomialFeatures(*args, **kwargs)


@scope.define
def sklearn_SplineTransformer(*args, **kwargs):
    return preprocessing.SplineTransformer(*args, **kwargs)


@validate(params=["order"],
          validation_test=lambda param: not isinstance(param, str) or param in ["C", "F"],
          msg="Invalid parameter '%s' with value '%s'. Value must be either 'C' or 'F'.")
def polynomial_features(name: str,
                        degree: typing.Union[int, tuple, Apply] = None,
                        interaction_only: typing.Union[bool, Apply] = None,
                        include_bias: typing.Union[bool, Apply] = None,
                        order: typing.Union[str, Apply] = None):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.preprocessing.PolynomialFeatures transformer.

    Args:
        name: name | str
        degree : degree of polynomial features | int
        interaction_only: produce only interaction features | bool
        include_bias: whether to include a bias column | bool
        order: order of output array in dense case | str
    """
    rval = scope.sklearn_PolynomialFeatures(
        degree=scope.int(hp.uniform(name + ".degree", 1, 5)) if degree is None else degree,
        interaction_only=hp.choice(name + ".interaction_only", [True, False])
        if interaction_only is None else interaction_only,
        include_bias=hp.choice(name + ".include_bias", [True, False]) if include_bias is None else include_bias,
        order=hp.choice(name + ".order", ["C", "F"]) if order is None else order
    )

    return rval


@validate(params=["knots"],
          validation_test=lambda param: not isinstance(param, str) or param in ["uniform", "quantile"],
          msg="Invalid parameter '%s' with value '%s'. Value must be either 'uniform' or 'quantile'.")
@validate(params=["extrapolation"],
          validation_test=lambda param: not isinstance(param, str) or param in ["error", "constant", "linear",
                                                                                "continue", "periodic"],
          msg="Invalid parameter '%s' with value '%s'. "
              "Value must be either 'error', 'constant', 'linear', 'continue', or 'periodic'.")
@validate(params=["order"],
          validation_test=lambda param: not isinstance(param, str) or param in ["C", "F"],
          msg="Invalid parameter '%s' with value '%s'. Value must be either 'C' or 'F'.")
def spline_transformer(name: str,
                       n_knots: typing.Union[int, Apply] = None,
                       degree: typing.Union[int, Apply] = None,
                       knots: typing.Union[str, npt.ArrayLike, Apply] = None,
                       extrapolation: typing.Union[str, Apply] = None,
                       include_bias: typing.Union[bool, Apply] = None,
                       order: typing.Union[str, Apply] = None):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.preprocessing.SplineTransformer transformer.

    Args:
        name: name | str
        n_knots: number of knots of splines | int
        degree: polynomial degree of spline | int
        knots: knot positions | str, npt.ArrayLike
        extrapolation: extrapolation method | str
        include_bias: whether to include a bias column | bool
        order: order of output array in dense case | str
    """
    rval = scope.sklearn_SplineTransformer(
        n_knots=scope.int(hp.uniform(name + ".n_knots", 5, 7)) if n_knots is None else n_knots,
        degree=scope.int(hp.uniform(name + ".degree", 2, 4)) if degree is None else degree,
        knots=hp.choice(name + ".knots", ["uniform", "quantile"]) if knots is None else knots,
        extrapolation=hp.choice(name + ".extrapolation", ["constant", "linear", "continue", "periodic"])
        if extrapolation is None else extrapolation,
        include_bias=hp.choice(name + ".include_bias", [True, False]) if include_bias is None else include_bias,
        order=hp.choice(name + ".order", ["C", "F"]) if order is None else order
    )

    return rval

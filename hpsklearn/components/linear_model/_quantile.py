from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import linear_model
import typing


@scope.define
def sklearn_QuantileRegressor(*args, **kwargs):
    return linear_model.QuantileRegressor(*args, **kwargs)


@validate(params=["solver"],
          validation_test=lambda param: not isinstance(param, str) or
          param in ["highs-ds", "highs-ipm", "highs", "revised simplex"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['highs-ds', 'highs-ipm', 'highs', "
              "'revised simplex'].")
def quantile_regression(name: str,
                        quantile: typing.Union[float, Apply] = None,
                        alpha: typing.Union[float, Apply] = None,
                        fit_intercept: typing.Union[bool, Apply] = None,
                        solver: typing.Union[str, Apply] = None,
                        solver_options: dict = None,
                        **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.QuantileRegression model.

    Args:
        name: name | str
        quantile: quantile the model tries to predict | float
        alpha: constant to multiply regularization term | float
        fit_intercept: whether to estimate intercept | bool
        solver: solver used for linear programming formulation | str
        solver_options: additional parameters for solver | dict
    """

    def _name(msg):
        return f"{name}.quantile_regression_{msg}"

    hp_space = dict(
        quantile=hp.normal(_name("quantile"), 0.5, 0.075) if quantile is None else quantile,
        alpha=hp.normal(_name("alpha"), mu=1.0, sigma=0.1) if alpha is None else alpha,
        fit_intercept=hp.choice(_name("fit_intercept"), [True, False]) if fit_intercept is None else fit_intercept,
        solver=hp.choice(_name("solver"), ["highs-ds", "highs-ipm", "highs", "revised simplex"])
        if solver is None else solver,
        solver_options=solver_options,
        **kwargs
    )

    return scope.sklearn_QuantileRegressor(**hp_space)

from hpsklearn.components._base import validate

from hyperopt.pyll import scope
from hyperopt import hp

from sklearn import linear_model


@scope.define
def sklearn_QuantileRegressor(*args, **kwargs):
    return linear_model.QuantileRegressor(*args, **kwargs)


@validate(params=["solver"],
          validation_test=lambda param: isinstance(param, str) and
                                        param in ["highs-ds", "highs-ipm", "highs",
                                                  "interior-point", "revised simplex"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['highs-ds', 'highs-ipm', 'highs', "
              "'interior-point', 'revised simplex'].")
def quantile_regression(name: str,
                        quantile: float = None,
                        alpha: float = None,
                        fit_intercept: bool = None,
                        solver: str = None,
                        solver_options: dict = None):
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
        alpha=alpha or hp.normal(_name("alpha"), mu=1.0, sigma=0.1),
        fit_intercept=fit_intercept or hp.choice(_name("fit_intercept"), [True, False]),
        solver=solver or hp.choice(_name("solver"),
                                   ["highs-ds", "highs-ipm", "highs", "interior-point", "revised simplex"]),
        solver_options=solver_options
    )

    return scope.sklearn_QuantileRegressor(**hp_space)

from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import discriminant_analysis
import numpy.typing as npt
import numpy as np
import typing


@scope.define
def sklearn_LinearDiscriminantAnalysis(*args, **kwargs):
    return discriminant_analysis.LinearDiscriminantAnalysis(*args, **kwargs)


@scope.define
def sklearn_QuadraticDiscriminantAnalysis(*args, **kwargs):
    return discriminant_analysis.QuadraticDiscriminantAnalysis(*args, **kwargs)


def _discriminant_analysis_tol(name: str):
    """
    Declaration search space 'tol' parameter
    """
    return hp.loguniform(name, np.log(1e-5), np.log(1e-2))


def _discriminant_analysis_hp_space(
        name_func,
        priors: npt.ArrayLike = None,
        store_covariance: bool = False,
        tol: float = None,
        **kwargs
):
    """
    Common hyper parameter search space
     linear discriminant analysis
     quadratic discriminant analysis
    """
    hp_space = dict(
        priors=priors,
        store_covariance=store_covariance,
        tol=_discriminant_analysis_tol(name_func("tol")) if tol is None else tol,
        **kwargs
    )
    return hp_space


@validate(params=["solver"],
          validation_test=lambda param: not isinstance(param, str) or param in ["svd", "lsqr", "eigen"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['svd', 'lsqr', 'eigen'].")
@validate(params=["shrinkage"],
          validation_test=lambda param: not isinstance(param, str) or param == "auto",
          msg="Invalid parameter '%s' with value '%s'. Value must be 'auto' or float.")
def linear_discriminant_analysis(name: str,
                                 solver: typing.Union[str, Apply] = None,
                                 shrinkage: typing.Union[float, str, Apply] = None,
                                 n_components: int = None,
                                 covariance_estimator: callable = None,
                                 **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.discriminant_analysis.LinearDiscriminantAnalysis model.

    Args:
        name: name | str
        solver: solver to use | str
        shrinkage: shrinkage parameter | str or float
        n_components: number of components | int
        covariance_estimator: covariance estimator to use | callable

    See help(hpsklearn.components.discriminant_analysis._discriminant_analysis_hp_space)
    for info on additional available discriminant analysis arguments.
    """

    def _name(msg):
        return f"{name}.linear_discriminant_analysis_{msg}"

    hp_space = _discriminant_analysis_hp_space(_name, **kwargs)
    hp_space["solver"] = hp.choice(_name("solver"), ["svd", "lsqr", "eigen"]) if solver is None else solver
    hp_space["shrinkage"] = shrinkage
    hp_space["n_components"] = n_components
    hp_space["covariance_estimator"] = covariance_estimator

    return scope.sklearn_LinearDiscriminantAnalysis(**hp_space)


def quadratic_discriminant_analysis(name: str, reg_param: typing.Union[float, Apply] = None, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis model.

    Args:
        name: name | str
        reg_param: regularization parameter | float

    See help(hpsklearn.components.discriminant_analysis._discriminant_analysis_hp_space)
    for info on additional available discriminant analysis arguments.
    """

    def _name(msg):
        return f"{name}.quadratic_discriminant_analysis_{msg}"

    hp_space = _discriminant_analysis_hp_space(_name, **kwargs)
    hp_space["reg_param"] = hp.uniform(_name("reg_param"), 0.0, 0.5) if reg_param is None else reg_param

    return scope.sklearn_QuadraticDiscriminantAnalysis(**hp_space)

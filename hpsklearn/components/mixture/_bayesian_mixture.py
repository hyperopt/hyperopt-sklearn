from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import mixture
import numpy.typing as npt
import numpy as np
import typing


@scope.define
def sklearn_BayesianGaussianMixture(*args, **kwargs):
    return mixture.BayesianGaussianMixture(*args, **kwargs)


@validate(params=["covariance_type"],
          validation_test=lambda param: not isinstance(param, str) or param in ["full", "tied", "diag", "spherical"],
          msg="Invalid parameter '%s' with value '%s'. Value must be one of 'full', 'tied', 'diag', or 'spherical'")
@validate(params=["init_params"],
          validation_test=lambda param: not isinstance(param, str) or param in ["kmeans", "random"],
          msg="Invalid parameter '%s' with value '%s'. Value must be one of 'kmeans' or 'random'")
@validate(params=["weight_concentration_prior_type"],
          validation_test=lambda param: not isinstance(param, str) or param in ["dirichlet_process",
                                                                                "dirichlet_distribution"],
          msg="Invalid parameter '%s' with value '%s'. Value must be 'dirichlet_process' or 'dirichlet_distribution'")
def bayesian_gaussian_mixture(name: str,
                              n_components: typing.Union[int, Apply] = None,
                              covariance_type: typing.Union[str, Apply] = None,
                              tol: typing.Union[float, Apply] = None,
                              reg_covar: typing.Union[float, Apply] = None,
                              max_iter: typing.Union[int, Apply] = None,
                              n_init: typing.Union[int, Apply] = None,
                              init_params: typing.Union[str, Apply] = None,
                              weight_concentration_prior_type: typing.Union[str, Apply] = None,
                              weight_concentration_prior: float = None,
                              mean_precision_prior: float = None,
                              mean_prior: npt.ArrayLike = None,
                              degrees_of_freedom_prior: float = None,
                              covariance_prior: typing.Union[npt.ArrayLike, float] = None,
                              random_state=None,
                              warm_start: bool = False,
                              verbose: int = 0,
                              verbose_interval: int = 10,
                              **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.mixture.BayesianGaussianMixture model.

    Args:
        name: name | str
        n_components: number of mixture components | int
        covariance_type: type of the covariance parameters | str
        tol: convergence threshold (stopping criterion) | float
        reg_covar: regularization added to the diagonal of covariance | float
        max_iter: maximum number of EM iterations | int
        n_init: number of initializations | int
        init_params: weights, means and covariance init method | str
        weight_concentration_prior_type: prior type for weight concentration | str
        weight_concentration_prior: dirichlet concentration | float
        mean_precision_prior: precision prior of mean distribution | float
        mean_prior: prior on the mean distribution | npt.ArrayLike
        degrees_of_freedom_prior: prior of degrees of freedom | float
        covariance_prior: prior on the covariance distribution | npt.ArrayLike
        random_state: random seed | int
        warm_start: reuse previous initialization if available | bool
        verbose: verbosity level | int
        verbose_interval: interval between log messages | int
    """

    def _name(msg):
        return f"{name}.bayesian_gaussian_mixture_{msg}"

    hp_space = dict(
        n_components=scope.int(hp.uniform(_name("n_components"), 1, 5)) if n_components is None else n_components,
        covariance_type=hp.choice(_name("covariance_type"), ["full", "tied", "diag", "spherical"])
        if covariance_type is None else covariance_type,
        tol=hp.loguniform(_name("tol"), np.log(1e-5), np.log(1e-2)) if tol is None else tol,
        reg_covar=hp.loguniform(_name("reg_covar"), np.log(1e-7), np.log(1e-5)) if reg_covar is None else reg_covar,
        max_iter=scope.int(hp.uniform(_name("max_iter"), 100, 300)) if max_iter is None else max_iter,
        n_init=hp.choice(_name("n_init"), [1, 2]) if n_init is None else n_init,
        init_params=hp.choice(_name("init_params"), ["kmeans", "random"]) if init_params is None else init_params,
        weight_concentration_prior_type=hp.choice(_name("weight_concentration_prior_type"), ["dirichlet_process",
                                                                                             "dirichlet_distribution"])
        if weight_concentration_prior_type is None else weight_concentration_prior_type,
        weight_concentration_prior=weight_concentration_prior,
        mean_precision_prior=mean_precision_prior,
        mean_prior=mean_prior,
        degrees_of_freedom_prior=degrees_of_freedom_prior,
        covariance_prior=covariance_prior,
        random_state=hp.randint(_name("random_state"), 5) if random_state is None else random_state,
        warm_start=warm_start,
        verbose=verbose,
        verbose_interval=verbose_interval,
        **kwargs,
    )
    return scope.sklearn_BayesianGaussianMixture(**hp_space)

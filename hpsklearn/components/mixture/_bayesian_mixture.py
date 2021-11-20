from hpsklearn.components._base import validate

from hyperopt.pyll import scope
from hyperopt import hp

from sklearn import mixture
import numpy.typing as npt
import numpy as np
import typing


@scope.define
def sklearn_BayesianGaussianMixture(*args, **kwargs):
    return mixture.BayesianGaussianMixture(*args, **kwargs)


@validate(params=["covariance_type"],
          validation_test=lambda param: param in ["full", "tied", "diag", "spherical"],
          msg="Invalid parameter '%s' with value '%s'. Value must be one of 'full', 'tied', 'diag', or 'spherical'")
@validate(params=["init_params"],
          validation_test=lambda param: param in ["kmeans", "random"],
          msg="Invalid parameter '%s' with value '%s'. Value must be one of 'kmeans' or 'random'")
@validate(params=["weight_concentration_prior_type"],
          validation_test=lambda param: param in ["dirichlet_process", "dirichlet_distribution"],
          msg="Invalid parameter '%s' with value '%s'. Value must be 'dirichlet_process' or 'dirichlet_distribution'")
def bayesian_gaussian_mixture(name: str,
                              n_components: int = None,
                              covariance_type: str = None,
                              tol: float = None,
                              reg_covar: float = None,
                              max_iter: int = None,
                              n_init: int = None,
                              init_params: str = None,
                              weight_concentration_prior_type: str = None,
                              weight_concentration_prior: float = None,
                              mean_precision_prior: float = None,
                              mean_prior: npt.ArrayLike = None,
                              degrees_of_freedom_prior: float = None,
                              covariance_prior: typing.Union[npt.ArrayLike, float] = None,
                              random_state=None,
                              warm_start: bool = False,
                              verbose: int = 0,
                              verbose_interval: int = 10):
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
        n_components=n_components or scope.int(hp.uniform(_name("n_components"), 1, 5)),
        covariance_type=covariance_type or hp.choice(_name("covariance_type"), ["full", "tied", "diag", "spherical"]),
        tol=hp.loguniform(_name("tol"), np.log(1e-5), np.log(1e-2)) if tol is None else tol,
        reg_covar=hp.loguniform(_name("reg_covar"), np.log(1e-7), np.log(1e-5)) if reg_covar is None else reg_covar,
        max_iter=max_iter or scope.int(hp.uniform(_name("max_iter"), 100, 300)),
        n_init=n_init or hp.choice(_name("n_init"), [1, 2]),
        init_params=init_params or hp.choice(_name("init_params"), ["kmeans", "random"]),
        weight_concentration_prior_type=weight_concentration_prior_type
        or hp.choice(_name("weight_concentration_prior_type"), ["dirichlet_process", "dirichlet_distribution"]),
        weight_concentration_prior=weight_concentration_prior,
        mean_precision_prior=mean_precision_prior,
        mean_prior=mean_prior,
        degrees_of_freedom_prior=degrees_of_freedom_prior,
        covariance_prior=covariance_prior,
        random_state=hp.randint(_name("random_state"), 5) if random_state is None else random_state,
        warm_start=warm_start,
        verbose=verbose,
        verbose_interval=verbose_interval,
    )
    return scope.sklearn_BayesianGaussianMixture(**hp_space)

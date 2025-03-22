from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import neural_network
import typing


@scope.define
def sklearn_MLPClassifier(*args, **kwargs):
    return neural_network.MLPClassifier(*args, **kwargs)


@scope.define
def sklearn_MLPRegressor(*args, **kwargs):
    return neural_network.MLPRegressor(*args, **kwargs)


def _multilayer_perceptron_activation(name: str):
    """
    Declaration search space 'activation' parameter
    """
    return hp.pchoice(name, [(0.2, "identity"), (0.2, "logistic"), (0.2, "tanh"), (0.4, "relu")])


def _multilayer_perceptron_solver(name: str):
    """
    Declaration search space 'solver' parameter
    """
    return hp.pchoice(name, [(0.2, "lbfgs"), (0.2, "sgd"), (0.6, "adam")])


def _multilayer_perceptron_alpha(name: str):
    """
    Declaration search space 'alpha' parameter
    """
    return hp.uniform(name, 1e-4, 0.01)


def _multilayer_perceptron_learning_rate(name: str):
    """
    Declaration search space 'learning_rate' parameter
    """
    return hp.choice(name, ["constant", "invscaling", "adaptive"])


def _multilayer_perceptron_learning_rate_init(name: str):
    """
    Declaration search space 'learning_rate_init' parameter
    """
    return hp.uniform(name, 1e-4, 0.1)


def _multilayer_perceptron_power_t(name: str):
    """
    Declaration search space 'power_t' parameter
    """
    return hp.uniform(name, 0.1, 0.9)


def _multilayer_perceptron_max_iter(name: str):
    """
    Declaration search space 'max_iter' parameter
    """
    return scope.int(hp.uniform(name, 150, 350))


def _multilayer_perceptron_random_state(name: str):
    """
    Declaration search space 'random_state' parameter
    """
    return hp.randint(name, 5)


def _multilayer_perceptron_tol(name: str):
    """
    Declaration search space 'tol' parameter
    """
    return hp.uniform(name, 1e-4, 0.01)


def _multilayer_perceptron_momentum(name: str):
    """
    Declaration search space 'momentum' parameter
    """
    return hp.uniform(name, 0.8, 1.0)


def _multilayer_perceptron_nesterovs_momentum(name: str):
    """
    Declaration search space 'nesterovs_momentum' parameter
    """
    return hp.choice(name, [True, False])


def _multilayer_perceptron_early_stopping(name: str):
    """
    Declaration search space 'early_stopping' parameter
    """
    return hp.choice(name, [True, False])


def _multilayer_perceptron_validation_fraction(name: str):
    """
    Declaration search space 'validation_fraction' parameter
    """
    return hp.uniform(name, 0.01, 0.2)


def _multilayer_perceptron_beta_1(name: str):
    """
    Declaration search space 'beta_1' parameter
    """
    return hp.uniform(name, 0.8, 1.0)


def _multilayer_perceptron_beta_2(name: str):
    """
    Declaration search space 'beta_2' parameter
    """
    return hp.uniform(name, 0.95, 1.0)


def _multilayer_perceptron_epsilon(name: str):
    """
    Declaration search space 'epsilon' parameter
    """
    return hp.uniform(name, 1e-9, 1e-5)


def _multilayer_perceptron_n_iter_no_change(name: str):
    """
    Declaration search space 'n_iter_no_change' parameter
    """
    return hp.choice(name, [10, 20, 30])


def _multilayer_perceptron_max_fun(name: str):
    """
    Declaration search space 'max_fun' parameter
    """
    return scope.int(hp.uniform(name, 1e4, 3e4))


@validate(params=["activation"],
          validation_test=lambda param: not isinstance(param, str) or param in ["identity", "logistic", "tanh", "relu"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['identity', 'logistic', 'tanh', 'relu'].")
@validate(params=["solver"],
          validation_test=lambda param: not isinstance(param, str) or param in ["lbfgs", "sgd", "adam"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['lbfgs', 'sgd', 'adam'].")
@validate(params=["learning_rate"],
          validation_test=lambda param: not isinstance(param, str) or param in ["constant", "invscaling", "adaptive"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['constant', 'invscaling', 'adaptive'].")
def _multilayer_perceptron_hp_space(
        name_func,
        hidden_layer_sizes: typing.Union[tuple, Apply] = None,
        activation: typing.Union[str, Apply] = None,
        solver: typing.Union[str, Apply] = None,
        alpha: typing.Union[float, Apply] = None,
        batch_size: typing.Union[int, Apply] = None,
        learning_rate: typing.Union[str, Apply] = None,
        learning_rate_init: typing.Union[float, Apply] = None,
        power_t: typing.Union[float, Apply] = None,
        max_iter: typing.Union[int, Apply] = None,
        shuffle: bool = True,
        random_state=None,
        tol: typing.Union[float, Apply] = None,
        verbose: bool = False,
        warm_start: bool = False,
        momentum: typing.Union[float, Apply] = None,
        nesterovs_momentum: typing.Union[bool, Apply] = True,
        early_stopping: typing.Union[bool, Apply] = False,
        validation_fraction: typing.Union[float, Apply] = None,
        beta_1: typing.Union[float, Apply] = None,
        beta_2: typing.Union[float, Apply] = None,
        epsilon: typing.Union[float, Apply] = None,
        n_iter_no_change: typing.Union[int, Apply] = None,
        max_fun: typing.Union[int, Apply] = None,
        **kwargs,
):
    """
    Hyper parameter search space for
     mlp classifier
     mlp regressor
    """
    hp_space = dict(
        hidden_layer_sizes=(100,) if hidden_layer_sizes is None else hidden_layer_sizes,
        activation=_multilayer_perceptron_activation(name_func("activation")) if activation is None else activation,
        solver=_multilayer_perceptron_solver(name_func("solver")) if solver is None else solver,
        alpha=_multilayer_perceptron_alpha(name_func("alpha")) if alpha is None else alpha,
        batch_size="auto" if batch_size is None else batch_size,
        learning_rate=_multilayer_perceptron_learning_rate(name_func("learning_rate"))
        if learning_rate is None else learning_rate,
        learning_rate_init=_multilayer_perceptron_learning_rate_init(name_func("learning_rate_init"))
        if learning_rate_init is None else learning_rate_init,
        power_t=_multilayer_perceptron_power_t(name_func("power_t")) if power_t is None else power_t,
        max_iter=_multilayer_perceptron_max_iter(name_func("max_iter")) if max_iter is None else max_iter,
        shuffle=shuffle,
        random_state=_multilayer_perceptron_random_state(name_func("random_state"))
        if random_state is None else random_state,
        tol=_multilayer_perceptron_tol(name_func("tol")) if tol is None else tol,
        verbose=verbose,
        warm_start=warm_start,
        momentum=_multilayer_perceptron_momentum(name_func("momentum")) if momentum is None else momentum,
        nesterovs_momentum=_multilayer_perceptron_nesterovs_momentum(name_func("nesterovs_momentum"))
        if nesterovs_momentum is None else nesterovs_momentum,
        early_stopping=_multilayer_perceptron_early_stopping(name_func("early_stopping"))
        if early_stopping is None else early_stopping,
        validation_fraction=_multilayer_perceptron_validation_fraction(name_func("validation_fraction"))
        if validation_fraction is None else validation_fraction,
        beta_1=_multilayer_perceptron_beta_1(name_func("beta_1")) if beta_1 is None else beta_1,
        beta_2=_multilayer_perceptron_beta_2(name_func("beta_2")) if beta_2 is None else beta_2,
        epsilon=_multilayer_perceptron_epsilon(name_func("epsilon")) if epsilon is None else epsilon,
        n_iter_no_change=_multilayer_perceptron_n_iter_no_change(name_func("n_iter_no_change"))
        if n_iter_no_change is None else n_iter_no_change,
        max_fun=_multilayer_perceptron_max_fun(name_func("max_fun")) if max_fun is None else max_fun,
        **kwargs,
    )
    return hp_space


def mlp_classifier(name: str, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.neural_network.MLPClassifier model.

    Args:
        name: name | str

    See help(hpsklearn.components.neural_network._multilayer_perceptron._multilayer_perceptron_hp_space)
    for info on additional available multilayer perceptron arguments.
    """

    def _name(msg):
        return f"{name}.mlp_classifier_{msg}"

    hp_space = _multilayer_perceptron_hp_space(_name, **kwargs)

    return scope.sklearn_MLPClassifier(**hp_space)


def mlp_regressor(name: str, **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.neural_network.MLPRegressor model.

    Args:
        name: name | str

    See help(hpsklearn.components.neural_network._multilayer_perceptron._multilayer_perceptron_hp_space)
    for info on additional available multilayer perceptron arguments.
    """

    def _name(msg):
        return f"{name}.mlp_regressor_{msg}"

    hp_space = _multilayer_perceptron_hp_space(_name, **kwargs)

    return scope.sklearn_MLPRegressor(**hp_space)

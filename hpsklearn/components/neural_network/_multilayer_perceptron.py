from hpsklearn.components._base import validate

from hyperopt.pyll import scope
from hyperopt import hp

from sklearn import neural_network


@scope.define
def sklearn_MLPClassifier(*args, **kwargs):
    return neural_network.MLPClassifier(*args, **kwargs)


@scope.define
def sklearn_MLPRegressor(*args, **kwargs):
    return neural_network.MLPRegressor(*args, **kwargs)


@validate(params=["activation"],
          validation_test=lambda param: isinstance(param, str) and param in ["identity", "logistic", "tanh", "relu"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['identity', 'logistic', 'tanh', 'relu'].")
@validate(params=["solver"],
          validation_test=lambda param: isinstance(param, str) and param in ["lbfgs", "sgd", "adam"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['lbfgs', 'sgd', 'adam'].")
@validate(params=["learning_rate"],
          validation_test=lambda param: isinstance(param, str) and param in ["constant", "invscaling", "adaptive"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['constant', 'invscaling', 'adaptive'].")
def _multilayer_perceptron_hp_space(
        name_func,
        hidden_layer_sizes: tuple = None,
        activation: str = None,
        solver: str = None,
        alpha: float = None,
        batch_size: int = None,
        learning_rate: str = None,
        learning_rate_init: float = None,
        power_t: float = None,
        max_iter: int = None,
        shuffle: bool = True,
        random_state=None,
        tol: float = None,
        verbose: bool = False,
        warm_start: bool = False,
        momentum: float = None,
        nesterovs_momentum: bool = True,
        early_stopping: bool = False,
        validation_fraction: float = None,
        beta_1: float = None,
        beta_2: float = None,
        epsilon: float = None,
        n_iter_no_change: int = None,
        max_fun: int = None,
):
    """
    Hyper parameter search space for
     mlp classifier
     mlp regressor
    """
    hp_space = dict(
        hidden_layer_sizes=(100,) if hidden_layer_sizes is None else hidden_layer_sizes,
        activation=activation or hp.pchoice(name_func("activation"), [(0.2, "identity"), (0.2, "logistic"),
                                                                      (0.2, "tanh"), (0.4, "relu")]),
        solver=solver or hp.pchoice(name_func("solver"), [(0.2, "lbfgs"), (0.2, "sgd"), (0.6, "adam")]),
        alpha=hp.uniform(name_func("alpha"), 1e-4, 0.01) if alpha is None else alpha,
        batch_size="auto" if batch_size is None else batch_size,
        learning_rate=learning_rate or hp.choice(name_func("learning_rate"), ["constant", "invscaling", "adaptive"]),
        learning_rate_init=hp.uniform(name_func("learning_rate_init"), 1e-4, 0.1)
        if learning_rate_init is None else learning_rate_init,
        power_t=hp.uniform(name_func("power_t"), 0.1, 0.9) if power_t is None else power_t,
        max_iter=max_iter or scope.int(hp.uniform(name_func("max_iter"), 150, 350)),
        shuffle=shuffle,
        random_state=hp.randint(name_func("random_state"), 5) if random_state is None else random_state,
        tol=hp.uniform(name_func("tol"), 1e-4, 0.01) if tol is None else tol,
        verbose=verbose,
        warm_start=warm_start,
        momentum=hp.uniform(name_func("momentum"), 0.8, 1.0) if momentum is None else momentum,
        nesterovs_momentum=nesterovs_momentum or hp.choice(name_func("nesterovs_momentum"), [True, False]),
        early_stopping=early_stopping or hp.choice(name_func("early_stopping"), [True, False]),
        validation_fraction=hp.uniform(name_func("validation_fraction"), 0.01, 0.2)
        if validation_fraction is None else validation_fraction,
        beta_1=hp.uniform(name_func("beta_1"), 0.8, 1.0) if beta_1 is None else beta_1,
        beta_2=hp.uniform(name_func("beta_2"), 0.95, 1.0) if beta_2 is None else beta_2,
        epsilon=hp.uniform(name_func("epsilon"), 1e-9, 1e-5) if epsilon is None else epsilon,
        n_iter_no_change=hp.choice(name_func("n_iter_no_change"), [10, 20, 30])
        if n_iter_no_change is None else n_iter_no_change,
        max_fun=scope.int(hp.uniform(name_func("max_fun"), 1e4, 3e4)) if max_fun is None else max_fun,
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

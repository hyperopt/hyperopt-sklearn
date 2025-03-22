import typing

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import covariance


@scope.define
def sklearn_EllipticEnvelope(*args, **kwargs):
    return covariance.EllipticEnvelope(*args, **kwargs)


def elliptic_envelope(name: str,
                      store_precision: bool = True,
                      assume_centered: bool = False,
                      support_fraction: typing.Union[float, Apply] = None,
                      contamination: typing.Union[float, Apply] = 0.1,
                      random_state=None,
                      **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.covariance.EllipticEnvelope model.

    Args:
        name: name | str
        store_precision: whether precision is stored | bool
        assume_centered: whether to assume centered data | bool
        support_fraction: fraction to include in support | float
        contamination: contamination of data set | float
        random_state: random state for shuffling data | int
    """

    def _name(msg):
        return f"{name}.elliptic_envelope_{msg}"

    hp_space = dict(
        store_precision=store_precision,
        assume_centered=assume_centered,
        support_fraction=hp.uniform(_name("support_fraction"), 0.05, 0.95)
        if support_fraction is None else support_fraction,
        contamination=hp.uniform(_name("contamination"), 0.0, 0.3) if contamination is None else contamination,
        random_state=hp.randint(_name("random_state"), 5) if random_state is None else random_state,
        **kwargs
    )
    return scope.sklearn_EllipticEnvelope(**hp_space)

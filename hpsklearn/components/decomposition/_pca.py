from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import decomposition
import numpy as np
import typing


@scope.define
def sklearn_PCA(*args, **kwargs):
    return decomposition.PCA(*args, **kwargs)


@validate(params=["n_components"],
          validation_test=lambda param: not isinstance(param, str) or param == "mle",
          msg="Invalid parameter '%s' with value '%s'. Choose 'mle', int or float.")
def pca(name: str,
        n_components: typing.Union[float, str, Apply] = None,
        whiten: typing.Union[bool, Apply] = None,
        copy: bool = True):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.decomposition.PCA transformer.

    Args:
        name: name | str
        n_components: number of components to keep | int
        whiten: whether to apply whitening | bool
        copy: whether to overwrite or copy | bool
    """
    rval = scope.sklearn_PCA(
        # -- qloguniform is missing a "scale" parameter so we
        #    lower the "high" parameter and multiply by 4 out front
        n_components=4 * scope.int(hp.qloguniform(name + ".n_components", low=np.log(0.51), high=np.log(30.5), q=1.0))
        if n_components is None else n_components,
        whiten=hp.choice(name + ".whiten", [True, False]) if whiten is None else whiten,
        copy=copy,
    )

    return rval

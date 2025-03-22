from hyperopt.pyll import scope

from sklearn import compose


@scope.define
def sklearn_TransformedTargetRegressor(*args, **kwargs):
    return compose.TransformedTargetRegressor(*args, **kwargs)


def transformed_target_regressor(name: str,
                                 regressor: object = None,
                                 transformer: object = None,
                                 func: callable = None,
                                 inverse_func: callable = None,
                                 check_inverse: bool = True,
                                 **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.compose.TransformedTargetRegressor model.

    Args:
        name: name | str
        regressor: regressor object | object
        transformer: estimator object | object
        func: function to apply to `y` before fit | callable
        inverse_func: function to apply to prediction | callable
        check_inverse: check whether inverse leads to original targets | bool
    """

    def _name(msg):
        return f"{name}.transformed_target_regressor_{msg}"

    # TODO: Try implementing np.exp and np.log | np.sqrt and np.square combinations
    hp_space = dict(
        regressor=regressor,
        transformer=transformer,
        func=func,
        inverse_func=inverse_func,
        check_inverse=check_inverse,
        **kwargs
    )
    return scope.sklearn_TransformedTargetRegressor(**hp_space)

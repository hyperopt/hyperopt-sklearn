from hyperopt.pyll import scope

from sklearn import linear_model


@scope.define
def sklearn_LinearRegression(*args, **kwargs):
    return linear_model.LinearRegression(*args, **kwargs)


def linear_regression(name: str,
                      fit_intercept: bool = True,
                      copy_X: bool = True,
                      n_jobs: int = 1,
                      positive: bool = False,
                      **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.LinearRegression model.

    Args:
        name: name | str
        fit_intercept: calculate intercept for model | bool
        copy_X: copy or overwrite X | bool
        n_jobs: number of jobs for computation | int
        positive: force coefficient to be positive | bool
    """

    def _name(msg):
        return f"{name}.linear_regression_{msg}"

    hp_space = dict(
        fit_intercept=fit_intercept,
        copy_X=copy_X,
        n_jobs=n_jobs,
        positive=positive,
        **kwargs
    )

    return scope.sklearn_LinearRegression(**hp_space)

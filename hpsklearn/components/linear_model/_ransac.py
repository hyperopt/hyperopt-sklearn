from hpsklearn.components._base import validate

from hyperopt.pyll import scope, Apply
from hyperopt import hp

from sklearn import linear_model
import numpy as np
import typing


@scope.define
def sklearn_RANSACRegressor(*args, **kwargs):
    return linear_model.RANSACRegressor(*args, **kwargs)


@validate(params=["loss"],
          validation_test=lambda param: not isinstance(param, str) or param in ["absolute_error", "squared_error"],
          msg="Invalid parameter '%s' with value '%s'. Value must be in ['absolute_error', 'squared_error'].")
def ransac_regression(name: str,
                      estimator=None,
                      min_samples: float = None,
                      residual_threshold: float = None,
                      is_data_valid: callable = None,
                      is_model_valid: callable = None,
                      max_trials: typing.Union[int, Apply] = None,
                      max_skips: typing.Union[int, Apply] = None,
                      stop_n_inliers: typing.Union[int, Apply] = None,
                      stop_score: typing.Union[float, Apply] = None,
                      stop_probability: typing.Union[float, Apply] = None,
                      loss: typing.Union[callable, str, Apply] = None,
                      random_state=None,
                      **kwargs):
    """
    Return a pyll graph with hyperparameters that will construct
    a sklearn.linear_model.RANSACRegressor model.

    Args:
        name: name | str
        estimator: base estimator object
        min_samples: minimum number of samples chosen | float
        residual_threshold: maximum residual | float
        is_data_valid: function called before model is fitted | callable
        is_model_valid: function called with estimated model | callable
        max_trials: maximum number of iterations sample selection | int
        max_skips: maximum skipped iterations due to finding zero inliers | int
        stop_n_inliers: stop iteration if this number of inliers found | int
        stop_score: stop iteration if score is greater equal to value | float
        stop_probability: confidence param of N-samples RANSAC | float
        loss: loss function to use | str
        random_state: random seed | int
    """

    def _name(msg):
        return f"{name}.ransac_regression_{msg}"

    hp_space = dict(
        estimator=estimator,
        min_samples=min_samples,  # default None fits linear model with X.shape[1] + 1
        residual_threshold=residual_threshold,
        is_data_valid=is_data_valid,
        is_model_valid=is_model_valid,
        max_trials=scope.int(hp.uniform(_name("max_trials"), 50, 150)) if max_trials is None else max_trials,
        max_skips=np.inf if max_skips is None else max_skips,
        stop_n_inliers=np.inf if stop_n_inliers is None else stop_n_inliers,
        stop_score=np.inf if stop_score is None else stop_score,
        stop_probability=hp.uniform(_name("stop_probability"), 0.90, 0.99)
        if stop_probability is None else stop_probability,
        loss=hp.choice(_name("loss"), ["absolute_error", "squared_error"]) if loss is None else loss,
        random_state=hp.randint(_name("random_state"), 5) if random_state is None else random_state,
        **kwargs
    )

    return scope.sklearn_RANSACRegressor(**hp_space)

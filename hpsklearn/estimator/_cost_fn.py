import hyperopt.rand

from ._utils import _NonFiniteFeature, _custom_handler
from ._transform import _transform_combine_XEX
from ._pfit import _pfit_until_convergence

from sklearn.model_selection import StratifiedShuffleSplit, \
    ShuffleSplit, \
    LeaveOneOut, \
    StratifiedKFold, \
    KFold, \
    GroupKFold, \
    PredefinedSplit
from sklearn.metrics import accuracy_score, r2_score

import numpy as np
import typing
import copy
import time


def _cost_fn(argd,
             X,
             y,
             EX_list: typing.Union[list, tuple] = None,
             valid_size: float = 0.2,
             n_folds: int = None,
             kfolds_group: typing.Union[list, np.ndarray] = None,
             shuffle: bool = False,
             random_state: typing.Union[int, np.random.Generator] = np.random.default_rng(),
             use_partial_fit: bool = False,
             info: callable = print,
             timeout: float = float("inf"),
             _conn=None,
             loss_fn: callable = None,
             continuous_loss_fn: bool = False,
             best_loss: float = None,
             n_jobs: int = 1):
    """
    Calculate the loss metric after performing model selection

    Args:
        X:
            Input variables

        y:
            Output variables

        EX_list: list, default is None
            List of exogenous datasets. Each must have the same number of
            samples as X.

        valid_size: float, default is 0.2
            The portion of the dataset used as the validation set. If
            cv_shuffle is False, always use the last samples as validation.

        n_folds: int, default is None
            When n_folds is not None, use K-fold cross-validation when
            n_folds > 2. Or, use leave-one-out cross-validation when
            n_folds = -1. For Group K-fold cross-validation, functions as
            `n_splits`.

        kfolds_group: list or ndarray, default is None
            When kfolds_group is not None, use Group K-fold cross-validation
            with the specified groups. The length of kfolds_group must be
            equal to the number of samples in X.

        shuffle: bool, default is False
            Whether to perform sample shuffling before splitting the
            data into training and validation sets.

        random_state: Generator, default is np.random.default_rng()
            The random state used to seed the cross-validation shuffling.

        use_partial_fit: bool, default is False
            If the learner support partial fit, it can be used for online
            learning. However, the whole train set is not split into mini
            batches here. The partial fit is used to iteratively update
            parameters on the whole train set. Early stopping is used to kill
            the training when the validation score stops improving.

        info: callable, default is print
            Callable to handle information with during the loss calculation
            process.

        timeout: float (seconds), or None for no timeout, default is None
            Kill trial evaluations after this many seconds.

        _conn: multiprocessing.connection. PipeConnection | Connection, default is None
            PipeConnection object connected to Pipe. Used to send rval to
            duplex PipeConnection object.

        loss_fn: callable, default is None
            A function that takes the arguments (y_target, y_prediction)
            and computes a loss value to be minimized. If no function is
            specified, '1.0 - accuracy_score(y_target, y_prediction)' is used
            for classification and '1.0 - r2_score(y_target, y_prediction)'
            is used for regression.

        continuous_loss_fn: boolean, default is False
            When true, the loss function is passed the output of
            predict_proba() as the second argument.  This is to facilitate the
            use of continuous loss functions like cross entropy or AUC.  When
            false, the loss function is given the output of predict().  If
            true, `classifier` and `loss_fn` must also be specified.

        best_loss: float, default is None
            Optimal loss value from all iterations of the optimization
            process.

        n_jobs: integer, default is 1
            Use multiple CPU cores when training estimators which support
            multiprocessing.
    """
    # scikit-learn needs a legacy `numpy.random.RandomState` RNG
    if isinstance(random_state, np.random.Generator):
        random_state_sklearn: typing.Union[int, np.random.RandomState] = np.random.RandomState(
            random_state.bit_generator
        )
    else:
        random_state_sklearn = random_state

    t_start = time.time()
    try:
        if "classifier" in argd:
            classifier = argd["classifier"]
            regressor = argd["regressor"]
            preprocessing = argd["preprocessing"]
            ex_pps_list = argd["ex_preprocs"]
        else:
            classifier = argd["model"]["classifier"]
            regressor = argd["model"]["regressor"]
            preprocessing = argd["model"]["preprocessing"]
            ex_pps_list = argd["model"]["ex_preprocs"]

        learner = classifier if classifier is not None else regressor
        is_classif = classifier is not None

        if hasattr(learner, "n_jobs"):
            learner.n_jobs = n_jobs

        untrained_learner = copy.deepcopy(learner)

        # Determine cross-validation iterator.
        if n_folds is not None:
            if n_folds == -1:
                info("Will use leave-one-out CV")
                cv_iter = LeaveOneOut().split(X)
            elif is_classif:
                info(f"Will use stratified K-fold CV with K: {n_folds} and Shuffle: {shuffle}")
                cv_iter = StratifiedKFold(n_splits=n_folds,
                                          shuffle=shuffle,
                                          random_state=random_state_sklearn
                                          ).split(X, y)
            else:
                if kfolds_group is not None:
                    info(f"Will use Group K-fold CV with K: {n_folds} and Shuffle: {shuffle}")
                    cv_iter = GroupKFold(n_splits=n_folds).split(X, y, kfolds_group)
                else:
                    info(f"Will use K-fold CV with K: {n_folds} and Shuffle: {shuffle}")
                    cv_iter = KFold(n_splits=n_folds,
                                    shuffle=shuffle,
                                    random_state=random_state_sklearn).split(X)
        else:
            if not shuffle:  # always choose the last samples.
                info(f"Will use the last {valid_size} portion of samples for validation")
                n_train = int(len(y) * (1 - valid_size))
                valid_fold = np.ones(len(y), dtype=np.int64)
                valid_fold[:n_train] = -1  # "-1" indicates train fold.

                cv_iter = PredefinedSplit(valid_fold).split()
            elif is_classif:
                info(f"Will use stratified shuffle-and-split with validation portion: {valid_size}")
                cv_iter = StratifiedShuffleSplit(1, test_size=valid_size,
                                                 random_state=random_state_sklearn
                                                 ).split(X, y)
            else:
                info(f"Will use shuffle-and-split with validation portion: {valid_size}")
                cv_iter = ShuffleSplit(n_splits=1, test_size=valid_size,
                                       random_state=random_state_sklearn).split(X)

        # Use cv_iter for cross-validation prediction.
        cv_y_pool = np.array([])
        cv_pred_pool = np.array([])
        cv_n_iters = np.array([])
        for train_index, valid_index in cv_iter:
            Xfit, Xval = X[train_index], X[valid_index]
            yfit, yval = y[train_index], y[valid_index]
            if EX_list is not None:
                _EX_list = [(EX[train_index], EX[valid_index])
                            for EX in EX_list]
                EXfit_list, EXval_list = zip(*_EX_list)
            else:
                EXfit_list = None
                EXval_list = None

            XEXfit, XEXval = _transform_combine_XEX(
                Xfit, info, preprocessing, Xval,
                EXfit_list, ex_pps_list, EXval_list
            )
            learner = copy.deepcopy(untrained_learner)
            info(f"Training learner {learner} on X/EX of dimension {XEXfit.shape}")

            if hasattr(learner, "partial_fit") and use_partial_fit:
                learner, n_iters = _pfit_until_convergence(
                    learner, is_classif, XEXfit, yfit, info,
                    best_loss=best_loss, XEXval=XEXval, yval=yval,
                    timeout=timeout, t_start=t_start
                )
            else:
                learner.fit(XEXfit, yfit)
                n_iters = None

            if learner is None:
                break
            cv_y_pool = np.append(cv_y_pool, yval)
            info(f"Scoring on X/EX validation of shape {XEXval.shape}")
            if continuous_loss_fn:
                cv_pred_pool = np.append(cv_pred_pool, learner.predict_proba(XEXval))
            else:
                cv_pred_pool = np.append(cv_pred_pool, learner.predict(XEXval))
            cv_n_iters = np.append(cv_n_iters, n_iters)
        else:  # all CV folds are exhausted.
            if loss_fn is None:
                if is_classif:
                    loss = 1 - accuracy_score(cv_y_pool, cv_pred_pool)
                    # -- squared standard error of mean
                    lossvar = (loss * (1 - loss)) / max(1, len(cv_y_pool) - 1)
                    info("OK trial with accuracy %.1f +- %.1f" % (
                        100 * (1 - loss),
                        100 * np.sqrt(lossvar))
                    )
                else:
                    loss = 1 - r2_score(cv_y_pool, cv_pred_pool)
                    lossvar = None  # variance of R2 is undefined.
                    info("OK trial with R2 score %.2e" % (1 - loss))
            else:
                # Use a user specified loss function
                loss = loss_fn(cv_y_pool, cv_pred_pool)
                lossvar = None
                info("OK trial with loss %.1f" % loss)

            t_done = time.time()
            rval = {
                "loss": loss,
                "loss_variance": lossvar,
                "learner": untrained_learner,
                "preprocs": preprocessing,
                "ex_preprocs": ex_pps_list,
                "status": hyperopt.STATUS_OK,
                "duration": t_done - t_start,
                "iterations": (cv_n_iters.max()
                               if (hasattr(learner, "partial_fit") and use_partial_fit)
                               else None),
            }
            rtype = "return"

        # The for loop exit with break, one fold did not finish running.
        if learner is None:
            t_done = time.time()
            rval = {
                "status": hyperopt.STATUS_FAIL,
                "failure": "Not enough time to finish training on all CV folds",
                "duration": t_done - t_start,
            }
            rtype = "return"

    except (_NonFiniteFeature,) as exc:
        print("Failing trial due to NaN in", str(exc))
        rval, rtype = _custom_handler(str_exc="",
                                      t_start=t_start,
                                      exc=exc)

    except (ValueError,) as exc:
        rval, rtype = _custom_handler(str_exc="k must be less than or equal to the number of training points",
                                      t_start=t_start,
                                      exc=exc)

    except (AttributeError,) as exc:
        print("Failing due to k_means_ weirdness")
        rval, rtype = _custom_handler(str_exc="'NoneType' object has no attribute 'copy'",
                                      t_start=t_start,
                                      exc=exc)

    except Exception as exc:
        rval = exc
        rtype = "raise"

        # -- return the result to calling process
    _conn.send((rtype, rval))

import numpy as np
import copy
import time
import typing

# Constants for partial_fit

# The partial_fit method will not be run if there is less than
# timeout * timeout_buffer number of seconds left before timeout
timeout_buffer = 0.05

# The minimum number of iterations of the partial_fit method that must be run
# before early stopping can kick in is min_n_iters
min_n_iters = 7

# After best_loss_cutoff_n_iters iterations have occurred, the training can be
# stopped early if the validation scores are far from the best scores
best_loss_cutoff_n_iters = 35

# Early stopping can occur when the best validation score of the earlier runs is
# greater than that of the later runs, tipping_pt_ratio determines the split
tipping_pt_ratio = 0.6

# Retraining will be done with all training data for retrain_fraction
# multiplied by the number of iterations used to train the original learner
retrain_fraction = 1.2


def _pfit_until_convergence(learner,
                            is_classif: bool,
                            XEXfit,
                            yfit,
                            info: callable = print,
                            max_iters: int = None,
                            best_loss: float = None,
                            XEXval=None,
                            yval=None,
                            timeout: float = None,
                            t_start: float = None) -> typing.Tuple[any, int]:
    """
    Do partial fitting until the convergence criterion is met

    Args:
        learner:
            Learner to train

        is_classif: bool
            Is learner a classifier algorithm

        XEXfit:
            Indices of X from combined endogenous and exogenous datasets.

        yfit:
            Indices of output variables from combined endogenous and
            exogenous datasets.

        info: callable, default is print
            Callable to handle information with during the loss calculation
            process.

        max_iters: int, default is None
            Maximum number of iterations as a convergence parameter

        best_loss: float, default is None
            Floating point containing the best loss value of all optimization
            iterations.

        XEXval:
            Values of X from combined endogenous and exogenous datasets.

        yval:
            Values of output variables from combined endogenous and exogenous
            datasets.

        timeout: float (seconds), or None for no timeout, default is None
            Kill trial evaluations after this many seconds.

        t_start: float (seconds), default is None
            Start time of loss metric calculation.
    """
    if max_iters is None:
        assert XEXval is not None and yval is not None and best_loss is not None
    if timeout is not None:
        assert t_start is not None
        timeout_tolerance = timeout * timeout_buffer
    else:
        timeout, t_start = float("Inf"), float("inf")
        timeout_tolerance = 0.

    n_iters = 0  # Keep track of the number of training iterations
    best_learner = None

    rng = np.random.default_rng(6665)
    train_idxs = rng.permutation(XEXfit.shape[0])
    validation_scores = []

    while not convergence_met(max_iters=max_iters,
                              n_iters=n_iters,
                              t_start=t_start,
                              timeout=timeout,
                              timeout_tolerance=timeout_tolerance,
                              yval=yval,
                              validation_scores=validation_scores,
                              info=info,
                              best_loss=best_loss):
        n_iters += 1
        rng.shuffle(train_idxs)
        if is_classif:
            learner.partial_fit(XEXfit[train_idxs], yfit[train_idxs],
                                classes=np.unique(yfit) if is_classif else None)
        if XEXval is not None:
            validation_scores.append(learner.score(XEXval, yval))
            if max(validation_scores) == validation_scores[-1]:
                best_learner = copy.deepcopy(learner)
            info("VSCORE", validation_scores[-1])

    if XEXval is None:
        return learner, n_iters
    else:
        return best_learner, n_iters


def should_stop(scores: list,
                info: callable = print,
                best_loss: float = None) -> bool:
    """
    Determine if partial fitting should be stopped

    Args:
        scores: list
            Validation scores

        info: callable, default is print
            Callable to handle information with during the loss calculation
            process.

        best_loss: float, default is None
            Optimal loss value from all iterations of the optimization
            process.
    """
    # TODO: possibly extend min_n_iters based on how close the current
    #      score is to the best score, up to some larger threshold
    if len(scores) < min_n_iters:
        return False
    tipping_pt = int(tipping_pt_ratio * len(scores))
    early_scores = scores[:tipping_pt]
    late_scores = scores[tipping_pt:]
    if max(early_scores) >= max(late_scores):
        info("Stopping early due to no improvement in late scores")
        return True

    # TODO: make this less confusing and possibly more accurate
    if len(scores) > best_loss_cutoff_n_iters and \
            max(scores) < 1 - best_loss and \
            3 * (max(late_scores) - max(early_scores)) < \
            1 - best_loss - max(late_scores):
        info("Stopping early due to best_loss cutoff criterion")
        return True
    return False


def convergence_met(max_iters: int,
                    n_iters: int,
                    t_start: float = None,
                    timeout: float = None,
                    timeout_tolerance: float = 0.,
                    yval=None,
                    validation_scores: list = None,
                    info: callable = print,
                    best_loss: float = None) -> bool:
    """
    Determine if convergence is met on partial fitting procedure

    Args:
        max_iters: int
            Maximum number of iterations as a convergence parameter

        n_iters: int
            Number of iterations in the partial fitting procedure
            performed while convergence criterion is not met

        t_start: float (seconds), default is None
            Start time of loss metric calculation.

        timeout: float (seconds), or None for no timeout, default is None
            Kill trial evaluations after this many seconds.

        timeout_tolerance: float (seconds)
            Timeout tolerance

        yval:
            Values of output variables from combined endogenous and exogenous
            datasets.

        validation_scores: list
            Validation scores

        info: callable, default is print
            Callable to handle information with during the loss calculation
            process.

        best_loss: float, default is None
            Optimal loss value from all iterations of the optimization
            process.
    """
    if max_iters is not None and n_iters >= max_iters:
        return True
    if time.time() - t_start >= timeout - timeout_tolerance:
        return True
    if yval is not None:
        return should_stop(scores=validation_scores,
                           info=info,
                           best_loss=best_loss)
    else:
        return False

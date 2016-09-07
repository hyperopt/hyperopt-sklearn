"""
"""
import pickle
import copy
from functools import partial
from multiprocessing import Process, Pipe
import time
from sklearn.base import BaseEstimator
from sklearn.cross_validation import KFold, StratifiedKFold, LeaveOneOut, \
                                     ShuffleSplit, StratifiedShuffleSplit, \
                                     PredefinedSplit
from sklearn.metrics import accuracy_score, r2_score
from sklearn.decomposition import PCA

# For backwards compatibility with older versions of hyperopt.fmin
import inspect

import numpy as np
import warnings

import hyperopt
import scipy.sparse

from . import components

# Constants for partial_fit

# The partial_fit method will not be run if there is less than
# timeout * timeout_buffer number of seconds left before timeout
timeout_buffer = 0.05

# The minimum number of iterations of the partial_fit method that must be run
# before early stopping can kick in is min_n_iters
min_n_iters = 7

# After best_loss_cutoff_n_iters iterations have occured, the training can be
# stopped early if the validation scores are far from the best scores
best_loss_cutoff_n_iters = 35

# Early stopping can occur when the best validation score of the earlier runs is
# greater than that of the later runs, tipping_pt_ratio determines the split
tipping_pt_ratio = 0.6

# Retraining will be done with all training data for retrain_fraction
# multiplied by the number of iterations used to train the original learner
retrain_fraction = 1.2


class NonFiniteFeature(Exception):
    """
    """

def transform_combine_XEX(Xfit, info, en_pps=[], Xval=None, 
                          EXfit_list=None, ex_pps_list=[], EXval_list=None):
    '''Transform endogenous and exogenous datasets and combine them into a 
    single dataset for training and testing.
    '''

    def run_preprocs(preprocessings, Xfit, Xval=None):
        '''Run all preprocessing steps in a pipeline
        '''
        for pp_algo in preprocessings:
            info('Fitting', pp_algo, 'to X of shape', Xfit.shape)
            if isinstance(pp_algo, PCA):
                n_components = pp_algo.get_params()['n_components']
                n_components = min([n_components] + list(Xfit.shape))
                pp_algo.set_params(n_components=n_components)
                info('Limited PCA n_components at', n_components)
            pp_algo.fit(Xfit)
            info('Transforming Xfit', Xfit.shape)
            Xfit = pp_algo.transform(Xfit)
            # np.isfinite() does not work on sparse matrices
            if not (scipy.sparse.issparse(Xfit) or \
                    np.all(np.isfinite(Xfit))):
              # -- jump to NonFiniteFeature handler below
                raise NonFiniteFeature(pp_algo)
            if Xval is not None:
                info('Transforming Xval', Xval.shape)
                Xval = pp_algo.transform(Xval)
                if not (scipy.sparse.issparse(Xval) or \
                        np.all(np.isfinite(Xval))):
                  # -- jump to NonFiniteFeature handler below
                    raise NonFiniteFeature(pp_algo)
        return (Xfit, Xval)

    # import ipdb; ipdb.set_trace()
    transformed_XEX_list = []
    en_pps = list(en_pps)
    ex_pps_list = list(ex_pps_list)
    if ex_pps_list == [] and EXfit_list is not None:
        ex_pps_list = [[]] * len(EXfit_list)
    xex_pps_list = [en_pps] + ex_pps_list
    if EXfit_list is None:
        EXfit_list = []
        assert EXval_list is None
        EXval_list = []
    elif EXval_list is None:
        EXval_list = [None] * len(EXfit_list)
    EXfit_list = list(EXfit_list)
    EXval_list = list(EXval_list)
    XEXfit_list = [Xfit] + EXfit_list
    XEXval_list = [Xval] + EXval_list
    for pps, dfit, dval in zip(xex_pps_list, XEXfit_list, XEXval_list):
        if pps != []:
            dfit, dval = run_preprocs(pps, dfit, dval)
        if dval is not None:
            transformed_XEX_list.append( (dfit, dval) )
        else:
            transformed_XEX_list.append(dfit)

    if Xval is None:
        XEXfit = np.concatenate(transformed_XEX_list, axis=1)
        return XEXfit
    else:
        XEXfit_list, XEXval_list = zip(*transformed_XEX_list)
        XEXfit = np.concatenate(XEXfit_list, axis=1)
        XEXval = np.concatenate(XEXval_list, axis=1)
        return (XEXfit, XEXval)

def pfit_until_convergence(learner, is_classif, XEXfit, yfit, info,
                           max_iters=None, best_loss=None,
                           XEXval=None, yval=None, 
                           timeout=None, t_start=None):
    '''Do partial fitting until the convergence criterion is met
    '''
    if max_iters is None:
        assert XEXval is not None and yval is not None and\
            best_loss is not None
    if timeout is not None:
        assert t_start is not None
    def should_stop(scores):
        # TODO: possibly extend min_n_iters based on how close the current
        #      score is to the best score, up to some larger threshold
        if len(scores) < min_n_iters:
            return False
        tipping_pt = int(tipping_pt_ratio * len(scores))
        early_scores = scores[:tipping_pt]
        late_scores = scores[tipping_pt:]
        if max(early_scores) >= max(late_scores):
            info("stopping early due to no improvement in late scores")
            return True
        # TODO: make this less confusing and possibly more accurate
        if len(scores) > best_loss_cutoff_n_iters and \
                max(scores) < 1 - best_loss and \
                3 * ( max(late_scores) - max(early_scores) ) < \
                1 - best_loss - max(late_scores):
            info("stopping early due to best_loss cutoff criterion")
            return True
        return False

    n_iters = 0  # Keep track of the number of training iterations
    best_learner = None
    if timeout is not None:
        timeout_tolerance = timeout * timeout_buffer
    else:
        timeout = float('Inf')
        timeout_tolerance = 0.
        t_start = float('Inf')
    rng = np.random.RandomState(6665)
    train_idxs = rng.permutation(XEXfit.shape[0])
    validation_scores = []

    def convergence_met():
        if max_iters is not None and n_iters >= max_iters:
            return True
        if time.time() - t_start >= timeout - timeout_tolerance:
            return True
        if yval is not None:
            return should_stop(validation_scores)
        else:
            return False

    while not convergence_met():
        n_iters += 1
        rng.shuffle(train_idxs)
        if is_classif:
            learner.partial_fit(XEXfit[train_idxs], yfit[train_idxs],
                                classes=np.unique(yfit))
        else:
            learner.partial_fit(XEXfit[train_idxs], yfit[train_idxs])
        if XEXval is not None:
            validation_scores.append(learner.score(XEXval, yval))
            if max(validation_scores) == validation_scores[-1]:
                best_learner = copy.deepcopy(learner)
            info('VSCORE', validation_scores[-1])    
    if XEXval is None:
        return (learner, n_iters)
    else:
        return (best_learner, n_iters)


def _cost_fn(argd, X, y, EX_list, valid_size, n_folds, shuffle, random_state,
             use_partial_fit, info, timeout, _conn, loss_fn=None, best_loss=None):
    '''Calculate the loss function
    '''
    try:
        t_start = time.time()
        # Extract info from calling function.
        if 'classifier' in argd:
            classifier = argd['classifier']
            regressor = argd['regressor']
            preprocessings = argd['preprocessing']
            ex_pps_list = argd['ex_preprocs']
        else:
            classifier = argd['model']['classifier']
            regressor = argd['model']['regressor']
            preprocessings = argd['model']['preprocessing']
            ex_pps_list = argd['model']['ex_preprocs']
        learner = classifier if classifier is not None else regressor
        is_classif = classifier is not None
        untrained_learner = copy.deepcopy(learner)
        # -- N.B. modify argd['preprocessing'] in-place

        # Determine cross-validation iterator.
        if n_folds is not None:
            if n_folds == -1:
                info('Will use leave-one-out CV')
                cv_iter = LeaveOneOut(len(y))
            elif is_classif:
                info('Will use stratified K-fold CV with K:', n_folds,
                     'and Shuffle:', shuffle)
                cv_iter = StratifiedKFold(y, n_folds=n_folds, 
                                          shuffle=shuffle, 
                                          random_state=random_state)
            else:
                info('Will use K-fold CV with K:', n_folds,
                     'and Shuffle:', shuffle)
                cv_iter = KFold(len(y), n_folds=n_folds, 
                                shuffle=shuffle, 
                                random_state=random_state)
        else:
            if not shuffle:  # always choose the last samples.
                info('Will use the last', valid_size, 
                     'portion of samples for validation')
                n_train = int(len(y) * (1 - valid_size))
                valid_fold = np.ones(len(y), dtype=np.int)
                valid_fold[:n_train] = -1  # "-1" indicates train fold.
                cv_iter = PredefinedSplit(valid_fold)
            elif is_classif:
                info('Will use stratified shuffle-and-split with validation \
                      portion:', valid_size)
                cv_iter = StratifiedShuffleSplit(y, 1, test_size=valid_size, 
                                                 random_state=random_state)
            else:
                info('Will use shuffle-and-split with validation portion:', 
                     valid_size)
                cv_iter = ShuffleSplit(len(y), 1, test_size=valid_size, 
                                       random_state=random_state)

        # Use the above iterator for cross-validation prediction.
        cv_y_pool = np.array([])
        cv_pred_pool = np.array([])
        cv_n_iters = np.array([])
        for train_index, valid_index in cv_iter:
            Xfit, Xval = X[train_index], X[valid_index]
            yfit, yval = y[train_index], y[valid_index]
            if EX_list is not None:
                _EX_list = [ (EX[train_index], EX[valid_index]) 
                             for EX in EX_list ]
                EXfit_list, EXval_list = zip(*_EX_list)
            else:
                EXfit_list = None
                EXval_list = None        
            XEXfit, XEXval = transform_combine_XEX(
                Xfit, info, preprocessings, Xval, 
                EXfit_list, ex_pps_list, EXval_list
            )
            learner = copy.deepcopy(untrained_learner)
            info('Training learner', learner, 'on X/EX of dimension', 
                 XEXfit.shape)
            if hasattr(learner, "partial_fit") and use_partial_fit:
                learner, n_iters = pfit_until_convergence(
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
            info('Scoring on X/EX validation of shape', XEXval.shape)
            cv_pred_pool = np.append(cv_pred_pool, learner.predict(XEXval))
            cv_n_iters = np.append(cv_n_iters, n_iters)
        else:  # all CV folds are exhausted.
            if loss_fn is None:
                if is_classif:
                    loss = 1 - accuracy_score(cv_y_pool, cv_pred_pool)
                    # -- squared standard error of mean
                    lossvar = (loss * (1 - loss)) / max(1, len(cv_y_pool) - 1)
                    info('OK trial with accuracy %.1f +- %.1f' % (
                         100 * (1 - loss),
                         100 * np.sqrt(lossvar))
                    )
                else:
                    loss = 1 - r2_score(cv_y_pool, cv_pred_pool)
                    lossvar = None  # variance of R2 is undefined.
                    info('OK trial with R2 score %.2e' % (1 - loss))
            else:
                # Use a user specified loss function
                loss = loss_fn(cv_y_pool, cv_pred_pool)
                lossvar = None
                info('OK trial with loss %.1f' % loss)
            t_done = time.time()
            rval = {
                'loss': loss,
                'loss_variance': lossvar,
                'learner': untrained_learner,
                'preprocs': preprocessings,
                'ex_preprocs': ex_pps_list,
                'status': hyperopt.STATUS_OK,
                'duration': t_done - t_start,
                'iterations': cv_n_iters.max(),
            }
            rtype = 'return'
        # The for loop exit with break, one fold did not finish running.
        if learner is None:
            t_done = time.time()
            rval = {
                'status': hyperopt.STATUS_FAIL,
                'failure': 'Not enough time to finish training on \
                            all CV folds',
                'duration': t_done - t_start,
            }
            rtype = 'return'

    ##==== Cost function exception handling ====##
    except (NonFiniteFeature,) as exc:
        print('Failing trial due to NaN in', str(exc))
        t_done = time.time()
        rval = {
            'status': hyperopt.STATUS_FAIL,
            'failure': str(exc),
            'duration': t_done - t_start,
        }
        rtype = 'return'

    except (ValueError,) as exc:
        if ('k must be less than or equal'
                ' to the number of training points') in str(exc):
            t_done = time.time()
            rval = {
                'status': hyperopt.STATUS_FAIL,
                'failure': str(exc),
                'duration': t_done - t_start,
            }
            rtype = 'return'
        else:
            rval = exc
            rtype = 'raise'

    except (AttributeError,) as exc:
        print('Failing due to k_means_ weirdness')
        if "'NoneType' object has no attribute 'copy'" in str(exc):
            # -- sklearn/cluster/k_means_.py line 270 raises this sometimes
            t_done = time.time()
            rval = {
                'status': hyperopt.STATUS_FAIL,
                'failure': str(exc),
                'duration': t_done - t_start,
            }
            rtype = 'return'
        else:
            rval = exc
            rtype = 'raise'

    except Exception as exc:
        rval = exc
        rtype = 'raise'

    # -- return the result to calling process
    _conn.send((rtype, rval))


class hyperopt_estimator(BaseEstimator):

    def __init__(self,
                 preprocessing=None,
                 ex_preprocs=None,
                 classifier=None,
                 regressor=None,
                 space=None,
                 algo=None,
                 max_evals=10,
                 loss_fn=None,
                 verbose=False,
                 trial_timeout=None,
                 fit_increment=1,
                 fit_increment_dump_filename=None,
                 seed=None,
                 use_partial_fit=False,
                 ):
        """
        Parameters
        ----------

        preprocessing: pyll.Apply node, default is None
            This should evaluate to a list of sklearn-style preprocessing
            modules (may include hyperparameters). When None, a random 
            preprocessing module will be used.

        ex_preprocs: pyll.Apply node, default is None
            This should evaluate to a list of lists of sklearn-style 
            preprocessing modules for each exogenous dataset. When None, no 
            preprocessing will be applied to exogenous data.

        classifier: pyll.Apply node
            This should evaluates to sklearn-style classifier (may include
            hyperparameters).

        regressor: pyll.Apply node
            This should evaluates to sklearn-style regressor (may include
            hyperparameters).

        algo: hyperopt suggest algo (e.g. rand.suggest)

        max_evals: int
            Fit() will evaluate up to this-many configurations. Does not apply
            to fit_iter, which continues to search indefinitely.

        loss_fn: callable
            A function that takes the arguments (y_target, y_prediction)
            and computes a loss value to be minimized. If no function is
            specified, '1.0 - accuracy_score(y_target, y_prediction)' is used
            for classification and '1.0 - r2_score(y_target, y_prediction)'
            is used for regression

        trial_timeout: float (seconds), or None for no timeout
            Kill trial evaluations after this many seconds.

        fit_increment: int
            Every this-many trials will be a synchronization barrier for
            ongoing trials, and the hyperopt Trials object may be
            check-pointed.  (Currently evaluations are done serially, but
            that might easily change in future to allow e.g. MongoTrials)

        fit_increment_dump_filename : str or None
            Periodically dump self.trials to this file (via cPickle) during
            fit()  Saves after every `fit_increment` trial evaluations.

        seed: numpy.random.RandomState or int or None
            If int, the integer will be used to seed a RandomState instance 
            for use in hyperopt.fmin. Use None to make sure each run is 
            independent. Default is None.

        use_partial_fit : boolean
            If the learner support partial fit, it can be used for online 
            learning. However, the whole train set is not split into mini 
            batches here. The partial fit is used to iteratively update 
            parameters on the whole train set. Early stopping is used to kill 
            the training when the validation score stops improving.
        """
        self.max_evals = max_evals
        self.loss_fn = loss_fn
        self.verbose = verbose
        self.trial_timeout = trial_timeout
        self.fit_increment = fit_increment
        self.fit_increment_dump_filename = fit_increment_dump_filename
        self.use_partial_fit = use_partial_fit
        if space is None:
            if classifier is not None:
                assert regressor is None
                self.classification = True
            else:
                assert regressor is not None
                self.classification = False
                # classifier = components.any_classifier('classifier')
            if preprocessing is None:
                preprocessing = components.any_preprocessing('preprocessing')
            else:
                # assert isinstance(preprocessing, (list, tuple))
                pass
            if ex_preprocs is None:
                ex_preprocs = []
            else:
                assert isinstance(ex_preprocs, (list, tuple))
                # assert all(
                #     isinstance(pps, (list, tuple)) for pps in ex_preprocs
                # )
            self.n_ex_pps = len(ex_preprocs)
            self.space = hyperopt.pyll.as_apply({
                'classifier': classifier,
                'regressor': regressor,
                'preprocessing': preprocessing,
                'ex_preprocs': ex_preprocs
            })
        else:
            assert classifier is None
            assert regressor is None
            assert preprocessing is None
            assert ex_preprocs is None
            # self.space = hyperopt.pyll.as_apply(space)
            self.space = space
            evaled_space = space.eval()
            if 'ex_preprocs' in evaled_space:
                self.n_ex_pps = len(evaled_space['ex_preprocs'])
            else:
                self.n_ex_pps = 0
                self.ex_preprocs = []

        if algo is None:
            self.algo = hyperopt.rand.suggest
        else:
            self.algo = algo

        if seed is not None:
            self.rstate = (np.random.RandomState(seed) 
                           if isinstance(seed, int) else seed)
        else:
            self.rstate = np.random.RandomState()

        # Backwards compatibility with older version of hyperopt
        self.seed = seed
        if 'rstate' not in inspect.getargspec(hyperopt.fmin).args:
            print("Warning: Using older version of hyperopt.fmin")

    def info(self, *args):
        if self.verbose:
            print(' '.join(map(str, args)))

    def fit_iter(self, X, y, EX_list=None, valid_size=.2, n_folds=None, 
                 cv_shuffle=False, warm_start=False,
                 random_state=np.random.RandomState(),
                 weights=None, increment=None):
        """Generator of Trials after ever-increasing numbers of evaluations
        """
        assert weights is None
        increment = self.fit_increment if increment is None else increment

        # len does not work on sparse matrices, so using shape[0] instead
        # shape[0] does not work on lists, so using len() for those
        if scipy.sparse.issparse(X):
            data_length = X.shape[0]
        else:
            data_length = len(X)
        if type(X) is list:
            X = np.array(X)
        if type(y) is list:
            y = np.array(y)

        if not warm_start:
            self.trials = hyperopt.Trials()
            self._best_loss = float('inf')
        else:
            assert hasattr(self, 'trials')
        # self._best_loss = float('inf')
        # This is where the cost function is used.
        fn = partial(_cost_fn,
                     X=X, y=y, EX_list=EX_list, 
                     valid_size=valid_size, n_folds=n_folds, 
                     shuffle=cv_shuffle, random_state=random_state,
                     use_partial_fit=self.use_partial_fit,
                     info=self.info,
                     timeout=self.trial_timeout,
                     loss_fn=self.loss_fn)

        # Wrap up the cost function as a process with timeout control.
        def fn_with_timeout(*args, **kwargs):
            conn1, conn2 = Pipe()
            kwargs['_conn'] = conn2
            th = Process(target=partial(fn, best_loss=self._best_loss),
                         args=args, kwargs=kwargs)
            th.start()
            if conn1.poll(self.trial_timeout):
                fn_rval = conn1.recv()
                th.join()
            else:
                self.info('TERMINATING DUE TO TIMEOUT')
                th.terminate()
                th.join()
                fn_rval = 'return', {
                    'status': hyperopt.STATUS_FAIL,
                    'failure': 'TimeOut'
                }

            assert fn_rval[0] in ('raise', 'return')
            if fn_rval[0] == 'raise':
                raise fn_rval[1]

            # -- remove potentially large objects from the rval
            #    so that the Trials() object below stays small
            #    We can recompute them if necessary, and it's usually
            #    not necessary at all.
            if fn_rval[1]['status'] == hyperopt.STATUS_OK:
                fn_loss = float(fn_rval[1].get('loss'))
                fn_preprocs = fn_rval[1].pop('preprocs')
                fn_ex_preprocs = fn_rval[1].pop('ex_preprocs')
                fn_learner = fn_rval[1].pop('learner')
                fn_iters = fn_rval[1].pop('iterations')
                if fn_loss < self._best_loss:
                    self._best_preprocs = fn_preprocs
                    self._best_ex_preprocs = fn_ex_preprocs
                    self._best_learner = fn_learner
                    self._best_loss = fn_loss
                    self._best_iters = fn_iters
            return fn_rval[1]

        while True:
            new_increment = yield self.trials
            if new_increment is not None:
                increment = new_increment
            
            #FIXME: temporary workaround for rstate issue #35
            #       latest hyperopt.fmin() on master does not match PyPI
            if 'rstate' in inspect.getargspec(hyperopt.fmin).args:
                hyperopt.fmin(fn_with_timeout,
                              space=self.space,
                              algo=self.algo,
                              trials=self.trials,
                              max_evals=len(self.trials.trials) + increment,
                              rstate=self.rstate,
                              # -- let exceptions crash the program,
                              #    so we notice them.
                              catch_eval_exceptions=False,
                              return_argmin=False, # -- in case no success so far
                             )
            else:
                if self.seed is None:
                    hyperopt.fmin(fn_with_timeout,
                                  space=self.space,
                                  algo=self.algo,
                                  trials=self.trials,
                                  max_evals=len(self.trials.trials) + increment,
                                 )
                else:
                    hyperopt.fmin(fn_with_timeout,
                                  space=self.space,
                                  algo=self.algo,
                                  trials=self.trials,
                                  max_evals=len(self.trials.trials) + increment,
                                  rseed=self.seed,
                                 )

    def retrain_best_model_on_full_data(self, X, y, EX_list=None,
                                        weights=None):
        if EX_list is not None:
            assert isinstance(EX_list, (list, tuple))
            assert len(EX_list) == self.n_ex_pps
        XEX = transform_combine_XEX(
            X, self.info, en_pps=self._best_preprocs, 
            EXfit_list=EX_list, ex_pps_list=self._best_ex_preprocs
        )

        self.info('Training learner', self._best_learner, 
                  'on X/EX of dimension', XEX.shape)
        if hasattr(self._best_learner, 'partial_fit') and \
                self.use_partial_fit:
            self._best_learner, _ = pfit_until_convergence(
                self._best_learner, self.classification, XEX, y, self.info,
                max_iters=int(self._best_iters * retrain_fraction)
            )
        else:
            self._best_learner.fit(XEX, y)

    def fit(self, X, y, EX_list=None, 
            valid_size=.2, n_folds=None, 
            cv_shuffle=False, warm_start=False,
            random_state=np.random.RandomState(),
            weights=None):
        """
        Search the space of learners and preprocessing steps for a good
        predictive model of y <- X. Store the best model for predictions.

        Args:
            EX_list ([list]): List of exogenous datasets. Each must has the 
                              same number of samples as X.
            valid_size ([float]): The portion of the dataset used as the 
                                  validation set. If cv_shuffle is False, 
                                  always use the last samples as validation.
            n_folds ([int]): When n_folds is not None, use K-fold cross-
                             validation when n_folds > 2. Or, use leave-one-out
                             cross-validation when n_folds = -1.
            cv_shuffle ([boolean]): Whether do sample shuffling before 
                                    splitting the data into train and valid 
                                    sets or not.
            warm_start ([boolean]): If warm_start, the estimator will start 
                                    from an existing sequence of trials.
            random_state: The random state used to seed the cross-validation 
                          shuffling.

        Notes:
            For classification problems, will always use the stratified version 
            of the K-fold cross-validation or shuffle-and-split.
        """
        if EX_list is not None:
            assert isinstance(EX_list, (list, tuple))
            assert len(EX_list) == self.n_ex_pps

        filename = self.fit_increment_dump_filename
        fit_iter = self.fit_iter(X, y, EX_list=EX_list,
                                 valid_size=valid_size,
                                 n_folds=n_folds,
                                 cv_shuffle=cv_shuffle,
                                 warm_start=warm_start,
                                 random_state=random_state,
                                 weights=weights,
                                 increment=self.fit_increment)
        next(fit_iter)
        adjusted_max_evals = (self.max_evals if not warm_start else 
                              len(self.trials.trials) + self.max_evals)
        while len(self.trials.trials) < adjusted_max_evals:
            increment = min(self.fit_increment,
                            adjusted_max_evals - len(self.trials.trials))
            fit_iter.send(increment)
            if filename is not None:
                with open(filename, 'wb') as dump_file:
                    self.info('---> dumping trials to', filename)
                    pickle.dump(self.trials, dump_file)

        self.retrain_best_model_on_full_data(X, y, EX_list, weights)

    def predict(self, X, EX_list=None):
        """
        Use the best model found by previous fit() to make a prediction.
        """
        if EX_list is not None:
            assert isinstance(EX_list, (list, tuple))
            assert len(EX_list) == self.n_ex_pps

        # -- copy because otherwise np.utils.check_arrays sometimes does not
        #    produce a read-write view from read-only memory
        if scipy.sparse.issparse(X):
            X = scipy.sparse.csr_matrix(X)
        else:
            X = np.array(X)
        XEX = transform_combine_XEX(
            X, self.info, en_pps=self._best_preprocs, 
            EXfit_list=EX_list, ex_pps_list=self._best_ex_preprocs
        )
        return self._best_learner.predict(XEX)

    def score(self, X, y, EX_list=None):
        """
        Return the score (accuracy or R2) of the learner on 
        a given set of data
        """
        if EX_list is not None:
            assert isinstance(EX_list, (list, tuple))
            assert len(EX_list) == self.n_ex_pps

        # -- copy because otherwise np.utils.check_arrays sometimes does not
        #    produce a read-write view from read-only memory
        if scipy.sparse.issparse(X):
            X = scipy.sparse.csr_matrix(X)
        else:
            X = np.array(X)
        XEX = transform_combine_XEX(
            X, self.info, en_pps=self._best_preprocs, 
            EXfit_list=EX_list, ex_pps_list=self._best_ex_preprocs
        )
        return self._best_learner.score(XEX, y)

    def best_model(self):
        """
        Returns the best model found by the previous fit()
        """
        return {'learner': self._best_learner,
                'preprocs': self._best_preprocs,
                'ex_preprocs': self._best_ex_preprocs}





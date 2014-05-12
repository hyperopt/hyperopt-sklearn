"""
"""
import cPickle
import copy
from functools import partial
from multiprocessing import Process, Pipe
import time

import numpy as np

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
# multiplied by the number of iterations used to train the original classifier
retrain_fraction = 1.2

class NonFiniteFeature(Exception):
    """
    """

def _cost_fn(argd, Xfit, yfit, Xval, yval, info, timeout,
             _conn, best_loss=None):
    try:
        t_start = time.time()
        if 'classifier' in argd:
            classifier = argd['classifier']
            preprocessings = argd['preprocessing']
        else:
            classifier = argd['model']['classifier']
            preprocessings = argd['model']['preprocessing']
        untrained_classifier = copy.deepcopy( classifier )
        # -- N.B. modify argd['preprocessing'] in-place
        for pp_algo in preprocessings:
            info('Fitting', pp_algo, 'to X of shape', Xfit.shape)
            pp_algo.fit(Xfit)
            info('Transforming fit and Xval', Xfit.shape, Xval.shape)
            Xfit = pp_algo.transform(Xfit)
            Xval = pp_algo.transform(Xval)
            
            # np.isfinite() does not work on sparse matrices
            if not (scipy.sparse.issparse(Xfit) or scipy.sparse.issparse(Xval)):
              if not (
                  np.all(np.isfinite(Xfit))
                  and np.all(np.isfinite(Xval))):
                  # -- jump to NonFiniteFeature handler below
                  raise NonFiniteFeature(pp_algo)

        info('Training classifier', classifier,
             'on X of dimension', Xfit.shape)


        def should_stop(scores):
          #TODO: possibly extend min_n_iters based on how close the current
          #      score is to the best score, up to some larger threshold
          if len(scores) < min_n_iters:
            return False
          tipping_pt = int(tipping_pt_ratio * len(scores))
          early_scores = scores[:tipping_pt]
          late_scores = scores[tipping_pt:]
          if max(early_scores) >= max(late_scores):
            return True
          #TODO: make this less confusing and possibly more accurate
          if len(scores) > best_loss_cutoff_n_iters and \
                 max(scores) < 1 - best_loss and \
                 3 * ( max(late_scores) - max(early_scores) ) < \
                 1 - best_loss - max(late_scores):
            info("stopping early due to best_loss cutoff criterion")
            return True
          return False

        n_iters = 0 # Keep track of the number of training iterations
        best_classifier = None
        if hasattr( classifier, "partial_fit" ):
          if timeout is not None:
            timeout_tolerance = timeout * timeout_buffer
          rng = np.random.RandomState(6665)
          train_idxs = rng.permutation(Xfit.shape[0])
          validation_scores = []
          
          while timeout is not None and \
                time.time() - t_start < timeout - timeout_tolerance:
            n_iters += 1
            rng.shuffle(train_idxs)
            classifier.partial_fit(Xfit[train_idxs], yfit[train_idxs],
                                   classes=np.unique( yfit ))
            validation_scores.append(classifier.score(Xval, yval))
            if max(validation_scores) == validation_scores[-1]:
              best_classifier = copy.deepcopy(classifier)
              
            if should_stop(validation_scores):
              break
            info('VSCORE', validation_scores[-1])
          classifier = best_classifier
        else:
          classifier.fit( Xfit, yfit )

        if classifier is None:
            t_done = time.time()
            rval = {
                'status': hyperopt.STATUS_FAIL,
                'failure': 'Not enough time to train anything',
                'duration': t_done - t_start,
                }
            rtype = 'return'
        else:
            info('Scoring on Xval of shape', Xval.shape)
            loss = 1.0 - classifier.score(Xval,yval)
            # -- squared standard error of mean
            lossvar = (loss * (1 - loss)) / max(1, len(yval) - 1)
            info('OK trial with accuracy %.1f +- %.1f' % (
                100 * (1.0 - loss),
                100 * np.sqrt(lossvar)))
            t_done = time.time()
            rval = {
                'loss': loss,
                'loss_variance': lossvar,
                'classifier': untrained_classifier,
                'preprocs': preprocessings,
                'status': hyperopt.STATUS_OK,
                'duration': t_done - t_start,
                'iterations': n_iters,
                }
            rtype = 'return'
        
    except (NonFiniteFeature,), exc:
        print 'Failing trial due to NaN in', str(exc)
        t_done = time.time()
        rval = {
            'status': hyperopt.STATUS_FAIL,
            'failure': str(exc),
            'duration': t_done - t_start,
            }
        rtype = 'return'

    except (ValueError,), exc:
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

    except (AttributeError,), exc:
        print 'Failing due to k_means_ weirdness'
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

    except Exception, exc:
        rval = exc
        rtype = 'raise'

    # -- return the result to calling process
    _conn.send((rtype, rval))


class hyperopt_estimator(object):
    def __init__(self,
                 preprocessing=None,
                 classifier=None,
                 space=None,
                 algo=None,
                 max_evals=100,
                 verbose=0,
                 trial_timeout=None,
                 fit_increment=1,
                 fit_increment_dump_filename=None,
                 seed=None,
                ):
        """
        Parameters
        ----------

        preprocessing: pyll.Apply node
            This should evaluate to a list of sklearn-style preprocessing
            modules (may include hyperparameters).

        classifier: pyll.Apply node
            This should evaluates to sklearn-style classifier (may include
            hyperparameters).

        algo: hyperopt suggest algo (e.g. rand.suggest)

        max_evals: int
            Fit() will evaluate up to this-many configurations. Does not apply
            to fit_iter, which continues to search indefinitely.

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
        """
        self.max_evals = max_evals
        self.verbose = verbose
        self.trial_timeout = trial_timeout
        self.fit_increment = fit_increment
        self.fit_increment_dump_filename = fit_increment_dump_filename
        if space is None:
            if classifier is None:
                classifier = components.any_classifier('classifier')
            if preprocessing is None:
                preprocessing = components.any_preprocessing('preprocessing')
            self.space = hyperopt.pyll.as_apply({
                'classifier': classifier,
                'preprocessing': preprocessing,
            })
        else:
            assert classifier is None
            assert preprocessing is None
            self.space = hyperopt.pyll.as_apply(space)

        if algo is None:
            self.algo=hyperopt.rand.suggest
        else:
            self.algo = algo

        if seed is not None:
            self.rstate = np.random.RandomState(seed)
        else:
            self.rstate = np.random.RandomState()

    def info(self, *args):
        if self.verbose:
            print ' '.join(map(str, args))

    def fit_iter(self, X, y, weights=None, increment=None):
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
        
        p = np.random.RandomState(123).permutation( data_length )
        n_fit = int(.8 * data_length)
        Xfit = X[p[:n_fit]]
        yfit = y[p[:n_fit]]
        Xval = X[p[n_fit:]]
        yval = y[p[n_fit:]]
        self.trials = hyperopt.Trials()
        self._best_loss = float('inf')
        fn=partial(_cost_fn,
                Xfit=Xfit, yfit=yfit,
                Xval=Xval, yval=yval,
                info=self.info,
                timeout=self.trial_timeout)
        self._best_loss = float('inf')

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
                fn_classif = fn_rval[1].pop('classifier')
                fn_iters = fn_rval[1].pop('iterations')
                if fn_loss < self._best_loss:
                    self._best_preprocs = fn_preprocs
                    self._best_classif = fn_classif
                    self._best_loss = fn_loss
                    self._best_iters = fn_iters
            return fn_rval[1]

        while True:
            new_increment = yield self.trials
            if new_increment is not None:
                increment = new_increment
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

    def retrain_best_model_on_full_data(self, X, y, weights=None):
        for pp_algo in self._best_preprocs:
            pp_algo.fit(X)
            X = pp_algo.transform(X * 1) # -- * 1 avoids read-only copy bug
        if hasattr(self._best_classif, 'partial_fit'):
          rng = np.random.RandomState(6665)
          train_idxs = rng.permutation(X.shape[0])
          for i in xrange(int(self._best_iters * retrain_fraction)):
            rng.shuffle(train_idxs)
            self._best_classif.partial_fit(X[train_idxs], y[train_idxs],
                                           classes=np.unique(y))
        else:
          self._best_classif.fit(X,y)


    def fit(self, X, y, weights=None):
        """
        Search the space of classifiers and preprocessing steps for a good
        predictive model of y <- X. Store the best model for predictions.
        """
        filename = self.fit_increment_dump_filename
        fit_iter = self.fit_iter(X, y,
                                 weights=weights,
                                 increment=self.fit_increment)
        fit_iter.next()
        while len(self.trials.trials) < self.max_evals:
            increment = min(self.fit_increment,
                            self.max_evals - len(self.trials.trials))
            fit_iter.send(increment)
            if filename is not None:
                with open(filename, 'wb') as dump_file:
                    self.info('---> dumping trials to', filename)
                    cPickle.dump(self.trials, dump_file)

        self.retrain_best_model_on_full_data(X, y, weights)

    def predict(self, X):
        """
        Use the best model found by previous fit() to make a prediction.
        """

        # -- copy because otherwise np.utils.check_arrays sometimes does not
        #    produce a read-write view from read-only memory
        if scipy.sparse.issparse(X):
          X = scipy.sparse.csr_matrix(X)
        else:
          X = np.array(X)

        for pp in self._best_preprocs:
            self.info("Transforming X of shape", X.shape)
            X = pp.transform(X)
        self.info("Predicting X of shape", X.shape)
        return self._best_classif.predict(X)

    def score( self, X, y ):
        """
        Return the accuracy of the classifier on a given set of data
        """
        # -- copy because otherwise np.utils.check_arrays sometimes does not
        #    produce a read-write view from read-only memory
        X = np.array(X)
        for pp in self._best_preprocs:
            self.info("Transforming X of shape", X.shape)
            X = pp.transform(X)
        self.info("Classifying X of shape", X.shape)
        return self._best_classif.score(X, y)
    
    def best_model( self ):
        """
        Returns the best model found by the previous fit()
        """
        return {'classifier': self._best_classif,
                'preprocs': self._best_preprocs}




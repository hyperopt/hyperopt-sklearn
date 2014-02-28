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
import sklearn.datasets.base

from . import components

TIMEOUT_TOLERANCE = 2

class NonFiniteFeature(Exception):
    """
    """

def _cost_fn(argd, Xfit, yfit, Xval, yval, info, timeout,
             _conn,
             best_loss=None, # TODO: use this for stopping criterion
             partial_timeout_tolerance=5.0): # -- seconds
    try:
        t_start = time.time()
        classifier = argd['classifier']
        # -- N.B. modify argd['preprocessing'] in-place
        for pp_algo in argd['preprocessing']:
            info('Fitting', pp_algo, 'to X of shape', Xfit.shape)
            pp_algo.fit(Xfit)
            info('Transforming fit and Xval', Xfit.shape, Xval.shape)
            Xfit = pp_algo.transform(Xfit)
            Xval = pp_algo.transform(Xval)
            
            # np.isfinite() does not work on sparse matrices
            if not scipy.sparse.issparse(Xfit) and not scipy.sparse.issparse(Xval):
              if not (
                  np.all(np.isfinite(Xfit))
                  and np.all(np.isfinite(Xval))):
                  # -- jump to NonFiniteFeature handler below
                  raise NonFiniteFeature(pp_algo)

        info('Training classifier', classifier,
             'on X of dimension', Xfit.shape)

        min_n_iters = 7#5
        best_loss_cutoff_n_iters = 35
        early_stopping_fraction = 0.2
        tipping_pt_ratio = 0.6
        rng = np.random.RandomState(6665)

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

        info("about to check for partial fit")
        
        if hasattr( classifier, "partial_fit" ):
          info("it has partial fit")
          #perm = rng.permutation(Xfit.shape[0])
          train_idxs = rng.permutation(Xfit.shape[0])
          n_valid = int(early_stopping_fraction * Xfit.shape[0])
          #valid_idxs = perm[:n_valid]
          #train_idxs = perm[n_valid:]
          validation_scores = []
          while time.time() - t_start < timeout - partial_timeout_tolerance:
            rng.shuffle(train_idxs)
            assert Xfit[train_idxs].shape[0] == len(train_idxs)
            assert Xfit[train_idxs].shape[1] == Xfit.shape[1]
            classifier.partial_fit(Xfit[train_idxs], yfit[train_idxs],
                                   classes=np.unique( yfit ))
            validation_scores.append(classifier.score(Xval, yval))
              #classifier.score(Xfit[valid_idxs], yfit[valid_idxs]))
            if max(validation_scores) == validation_scores[-1]:
              best_classifier = copy.deepcopy(classifier)
              
            if should_stop(validation_scores):
              break
            #info('VSCORES', validation_scores)
            info('VSCORE', validation_scores[-1])
          classifier = best_classifier

#          # Split data into minibatches
#          #TODO: Shuffle the data - get a random index list of size num_splits
#          num_splits = int(Xfit.shape[0] / 100)
#          #Xpartial = np.array_split(Xfit, num_splits)
#          #ypartial = np.array_split(yfit, num_splits)
#
#          #FIXME: this will break if less than 100 points
#          #       a cleaner method should be used. Unfortunately a lot of clean
#          #       stuff doesn't work on sparse matrices
#          classifier.partial_fit( Xfit[0:100], yfit[0:100], classes=np.unique( yfit ) )
#          for i in range(1, num_splits-1):
#            classifier.partial_fit( Xfit[i*100:(i+1)*100],
#                                    yfit[i*100:(i+1)*100] )
#            if time.time() - t_start > timeout - TIMEOUT_TOLERANCE:
#              break
#            #TODO: do a predict and score here, and if it flattens out, break
#          #TODO: include the last bit of data in a cleaner way here
#          if time.time() - t_start < timeout - TIMEOUT_TOLERANCE:
#            classifier.partial_fit( Xfit[num_splits*100:],
#                                    yfit[num_splits*100:] )
        else:
          classifier.fit( Xfit, yfit )
        info('Scoring on Xval of shape', Xval.shape)
        loss = 1.0 - classifier.score(Xval, yval)
        info('OK trial with accuracy %.1f' % (100 * (1.0 - loss)))
        t_done = time.time()
        rval = {
            'loss': loss,
            'classifier': classifier,
            'preprocs': argd['preprocessing'],
            'status': hyperopt.STATUS_OK,
            'duration': t_done - t_start,
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
        self.verbose = 1 #verbose #TEMP
        self.trial_timeout = trial_timeout
        self.fit_increment = fit_increment
        self.fit_increment_dump_filename = fit_increment_dump_filename
        if algo is None:
            self.algo=hyperopt.rand.suggest
        else:
            self.algo = algo
        if classifier is None:
            classifier = components.any_classifier('classifier')

        self.classifier = classifier

        if preprocessing is None:
            preprocessing = components.any_preprocessing('preprocessing')

        self.preprocessing = preprocessing

        self.space = hyperopt.pyll.as_apply({
            'classifier': self.classifier,
            'preprocessing': self.preprocessing,
        })

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
                print 'TERMINATING DUE TO TIMEOUT'
                th.terminate()
                th.join()
                fn_rval = 'return', {
                    'status': hyperopt.STATUS_FAIL,
                    'failure': 'TimeOut'
                }

            assert fn_rval[0] in ('raise', 'return')
            if fn_rval[0] == 'raise':
                raise fn_rval[1]
                #"""
                fn_rval = 'raise', {
                    'status': hyperopt.STATUS_FAIL,
                    'failure': fn_rval[1]
                }
                return fn_rval[1]
                #"""

            # -- remove potentially large objects from the rval
            #    so that the Trials() object below stays small
            #    We can recompute them if necessary, and it's usually
            #    not necessary at all.
            if fn_rval[1]['status'] == hyperopt.STATUS_OK:
                fn_loss = float(fn_rval[1].get('loss'))
                fn_preprocs = fn_rval[1].pop('preprocs')
                fn_classif = fn_rval[1].pop('classifier')
                if fn_loss < self._best_loss:
                    self._best_preprocs = fn_preprocs
                    self._best_classif = fn_classif
                    self._best_loss = fn_loss
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
                         )


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
        # -- XXX: retrain best model on full data
        #print argmin

    def predict(self, X):
        """
        Use the best model found by previous fit() to make a prediction.
        """
        classifier = self._best_classif
        preprocs = self._best_preprocs

        # -- copy because otherwise np.utils.check_arrays sometimes does not
        #    produce a read-write view from read-only memory
        if scipy.sparse.issparse(X):
          X = scipy.sparse.csr_matrix(X)
        else:
          X = np.array(X)

        for pp in preprocs:
            X = pp.transform(X)
        return classifier.predict(X)

    def score( self, X, y ):
        """
        Return the accuracy of the classifier on a given set of data
        """
        classifier = self._best_classif
        preprocs = self._best_preprocs
        # -- copy because otherwise np.utils.check_arrays sometimes does not
        #    produce a read-write view from read-only memory
        X = np.array(X)
        for pp in preprocs:
            X = pp.transform(X)
        return classifier.score(X, y)
    
    def best_model( self ):
        """
        Returns the best model found by the previous fit()
        """
        #best_trial = self.trials.best_trial
        #return { 'classifier' : best_trial['result']['classifier'],
        #         'preprocs' : best_trial['result']['preprocs'] }
        return { 'classifier' : self._best_classif,
                 'preprocs' : self._best_preprocs }




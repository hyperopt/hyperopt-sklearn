"""
"""
import numpy as np
from functools import partial
import hyperopt
import components
from .thread2 import Thread

class TimeoutError(Exception):
    pass


class NonFiniteFeature(Exception):
    """
    """

def _cost_fn(argd, Xfit, yfit, Xval, yval, info, _rval):
    try:
        classifier = argd['classifier']
        # -- N.B. modify argd['preprocessing'] in-place
        for pp_algo in argd['preprocessing']:
            info('Fitting', pp_algo, 'to X of shape', Xfit.shape)
            pp_algo.fit(Xfit)
            info('Transforming fit and Xval', Xfit.shape, Xval.shape)
            Xfit = pp_algo.transform(Xfit)
            Xval = pp_algo.transform(Xval)
            if not (
                np.all(np.isfinite(Xfit))
                and np.all(np.isfinite(Xval))):
                # -- jump to NonFiniteFeature handler below
                raise NonFiniteFeature(pp_algo)

        info('Training classifier', classifier,
             'on X of dimension', Xfit.shape)
        classifier.fit(Xfit, yfit)
        info('Scoring on Xval of shape', Xval.shape)
        loss = 1.0 - classifier.score(Xval, yval)
        info('OK trial with accuracy %.1f' % (100 * (1 - loss)))
        rval = {
            'loss': loss,
            'classifier': classifier,
            'preprocs': argd['preprocessing'],
            'status': hyperopt.STATUS_OK,
            }
        _rval[0] = rval

    except (NonFiniteFeature,), exc:
        print 'Failing trial due to NaN in', str(exc)
        rval = {
            'status': hyperopt.STATUS_FAIL,
            'failure': str(exc),
            }
        _rval[0] = rval

    except (AttributeError,), exc:
        if "'NoneType' object has no attribute 'copy'" in str(exc):
            # -- sklearn/cluster/k_means_.py line 270 raises this sometimes
            rval = {
                'status': hyperopt.STATUS_FAIL,
                'failure': str(exc),
                }
        else:
            rval[:] = (exc, 'raise')
        _rval[0] = rval

    except (TimeoutError,), exc:
        rval = {
            'status': hyperopt.STATUS_FAIL,
            'failure': 'Timeout',
            }
        _rval[0] = rval

    except Exception, exc:
        rval[:] = (exc, 'raise')



class hyperopt_estimator(object):
    def __init__(self,
                 preprocessing=None,
                 classifier=None,
                 algo=None,
                 max_evals=100,
                 verbose=0,
                 fit_timeout=None):
        self.max_evals = max_evals
        self.verbose = verbose
        self.fit_timeout = fit_timeout
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

    def info(self, *args):
        if self.verbose:
            print ' '.join(map(str, args))

    def fit(self, X, y, weights=None):
        """
        Search the space of classifiers and preprocessing steps for a good
        predictive model of y <- X. Store the best model for predictions.
        """
        p = np.random.RandomState(123).permutation(len(X))
        n_fit = int(.8 * len(X))
        Xfit = X[p[:n_fit]]
        yfit = y[p[:n_fit]]
        Xval = X[p[n_fit:]]
        yval = y[p[n_fit:]]
        self.trials = hyperopt.Trials()
        fn=partial(_cost_fn,
                Xfit=Xfit, yfit=yfit,
                Xval=Xval, yval=yval,
                info=self.info)

        def fn_with_timeout(*args, **kwargs):
            kwargs['_rval'] = [None, None]
            th = Thread(target=fn, args=args, kwargs=kwargs)
            th.start()
            th.join(timeout=self.fit_timeout)
            if th.is_alive():
                th.terminate(TimeoutError)
                th.join()  # -- wait for other thread to pick up the exception and exit

            # -- assert _cost_fn exited in a controlled way
            assert kwargs['_rval'][0] != None

            if kwargs['_rval'][1] == 'raise':
                raise kwargs['_rval'][0]
            return kwargs['_rval'][0]

        hyperopt.fmin(fn_with_timeout,
            space=self.space,
            algo=self.algo,
            trials=self.trials,
            max_evals=self.max_evals)
        # -- XXX: retrain best model on full data
        #print argmin

    def predict(self, X):
        """
        Use the best model found by previous fit() to make a prediction.
        """
        best_trial = self.trials.best_trial
        classifier = best_trial['result']['classifier']
        preprocs = best_trial['result']['preprocs']

        # -- copy because otherwise np.utils.check_arrays sometimes does not
        #    produce a read-write view from read-only memory
        X = np.array(X)
        for pp in preprocs:
            X = pp.transform(X)
        return classifier.predict(X)

    def score( self, X, y ):
        """
        Return the accuracy of the classifier on a given set of data
        """
        best_trial = self.trials.best_trial
        classifier = best_trial['result']['classifier']
        preprocs = best_trial['result']['preprocs']
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
        best_trial = self.trials.best_trial
        return { 'classifier' : best_trial['result']['classifier'],
                 'preprocs' : best_trial['result']['preprocs'] }




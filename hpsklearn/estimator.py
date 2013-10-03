"""
"""
import numpy as np
from functools import partial
import hyperopt
import components

class NonFiniteFeature(Exception):
    """
    """

def _cost_fn(argd, Xfit, yfit, Xval, yval):
    try:
        classifier = argd['classifier']
        # -- N.B. modify argd['preprocessing'] in-place
        for pp_algo in argd['preprocessing']:
            pp_algo.fit(Xfit)
            Xfit = pp_algo.transform(Xfit)
            Xval = pp_algo.transform(Xval)
            if not (
                np.all(np.isfinite(Xfit))
                and np.all(np.isfinite(Xval))):
                raise NonFiniteFeature(pp_algo)

        classifier.fit(Xfit, yfit)
        loss = 1.0 - classifier.score(Xval, yval)
        print 'OK trial with accuracy %.1f'  % (100 * (1 - loss))
        rval = {
            'loss': loss,
            'classifier': classifier,
            'preprocs': argd['preprocessing'],
            'status': hyperopt.STATUS_OK,
            }
        return rval

    except (NonFiniteFeature,), exc:
        print 'Failing trial due to NaN in', str(exc)
        rval = {
            'loss': None,
            'status': hyperopt.STATUS_FAIL,
            'failure': str(exc),
            }

    except (AttributeError,), exc:
        if "'NoneType' object has no attribute 'copy'" in str(exc):
            # -- sklearn/cluster/k_means_.py line 270 raises this sometimes
            rval = {
                'loss': None,
                'status': hyperopt.STATUS_FAIL,
                'failure': str(exc),
                }
        else:
            raise
        return rval


class hyperopt_estimator(object):
    def __init__(self,
        preprocessing=None,
        classifier=None,
        algo=None,
        max_evals=100):
        self.max_evals = max_evals
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
        argmin = hyperopt.fmin(
            fn=partial(_cost_fn,
                Xfit=Xfit, yfit=yfit,
                Xval=Xval, yval=yval),
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
        for pp in preprocs:
            X = pp.transform(X)
        return classifier.predict(X)




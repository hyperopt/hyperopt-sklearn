"""
"""
import numpy as np
from functools import partial
import hyperopt
import components

def _cost_fn(argd, Xfit, yfit, Xval, yval):
    classifier = argd['classifier']
    # -- N.B. modify argd['preprocessing'] in-place
    for pp_algo in argd['preprocessing']:
        pp_algo.fit(Xfit)
        Xfit = pp_algo.transform(Xfit)
        Xval = pp_algo.transform(Xval)
    classifier.fit(Xfit, yfit)
    loss = -classifier.score(Xval, yval)
    rval = {
        'loss': loss,
        'classifier': classifier,
        'preprocs': argd['preprocessing'],
        'status': 'ok',
        }
    return rval


class hyperopt_estimator(object):
    def __init__(self, preprocessing=None, classifier=None,
            max_evals=100):
        self.max_evals = max_evals
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
            algo=hyperopt.rand.suggest,
            trials=self.trials,
            max_evals=self.max_evals)
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




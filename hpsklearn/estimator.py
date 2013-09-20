"""
"""
from functools import partial
import hyperopt
import components

def _cost_fn(argd, X, y):
    classifier = argd['classifier']
    preprocessing = argd['preprocessing']
    preprocessing.fit(X)
    X2 = preprocessing.transform(X)
    classifier.fit(X2, y)
    return classifier.score(X2, y)


class HyperoptEstimatorFactory(object):
    def __init__(self, preprocessing=None, classifier=None):
        if classifier is None:
            classifier = components.any_classifier()

        self.classifier = classifier

        if preprocessing is None:
            preprocessing = components.any_preprocessing()

        self.preprocessing = preprocessing

        self.space = hyperopt.pyll.as_apply({
            'classifier': self.classifier,
            'preprocessing': self.preprocessing,
        })

    def __call__(self, X):
        return self.predict(X)

    def fit(self, X, y, weights=None):
        """
        Search the space of classifiers and preprocessing steps for a good
        predictive model of y <- X. Store the best model for predictions.
        """
        trials = hyperopt.Trials()
        hyperopt.fmin(
            fn=partial(_cost_fn, X=X, y=y),
            space=self.space,
            algo=hyperopt.rand.suggest,
            trails=trials,
            max_iters=10)

    def predict(self, X):
        """
        Use the best model found by previous fit() to make a prediction.
        """
        raise NotImplementedError()


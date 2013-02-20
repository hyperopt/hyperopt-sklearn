import numpy as np
import hyperopt


def classification_scoring(y_val, y_hat):
    return np.mean(y_hat != y_val)


class HyperoptMetaEstimator(object):
    """Hyperopt search on the parameters of an estimator

    Important members are fit, predict.

    HyperoptCV implements a "fit" method and a "predict" method like
    any classifier except that the parameters of the classifier
    used to predict is optimized by cross-validation.

    This meta-estimator is supposed to be similar to GridSearchCV.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        A object of that type is instantiated for each grid point.

    search_space : fn
        Callable that returns a pyll search space, suitable for hyperopt to search.
        A sample from the search space should be a dictionary that can be used as kwargs
        to the estimator.

    scoring : string or callable, optional
        Either one of either a string ("zero_one", "f1", "roc_auc", ... for
        classification, "mse", "r2",... for regression) or a callable.
        See 'Scoring objects' in the model evaluation section of the user guide
        for details.


    """
    def __init__(self, estimator, search_space, scoring=None,
                 n_valid=0.2,
                 algo=hyperopt.tpe.suggest,
                 max_evals=100,
                ):
        """
        n_valid - float or int
            If float, the fraction of training data to use for model selection
            If int, the number of training data to use for model selection

        algo - a hyperopt search algorithm (see hyperopt.fmin)

        max_evals - int
            The maximum number of calls to `Perceptron.fit()` per call to
            `AutoPerceptron.fit()`

        search_space - fn
            An alternative to the `search_space` function above, which may
            re-parameterize the Perceptron configuration space. It will be
            called with a `make_name` argument, and nothing more.

        """
        self.estimator = estimator
        self.search_space = search_space
        self.scoring = scoring
        self.n_valid = n_valid
        self.algo = algo
        self.max_evals = max_evals
        self.search_space = search_space

    def fit(self, X, y):
        """
        X - matrix of features
        y - vector of integer labels

        See sklearn.linear_model.Perceptron.fit for more information.
        """
        err_rval = 1.0 # worst possible classification accuracy
        if 0 < self.n_valid < 1:
            n_valid = int(self.n_valid * len(X))
        else:
            n_valid = self.n_valid
        X_fit = X[n_valid:]
        y_fit = y[n_valid:]
        X_val = X[:n_valid]
        y_val = y[:n_valid]
        space = self.search_space(str)
        def eval_params(kw):
            model = self.estimator(**kw)
            try:
                model.fit(X_fit, y_fit)
            except ValueError, e:
                if 'overflow' in str(e):
                    # This can happen once in a while, maybe due to large
                    # learning rate? Or large regularization?
                    return err_rval
                raise
            y_hat = model.predict(X_val)
            val_erate = self.scoring(y_val, y_hat)
            return {'loss': val_erate,
                    'status': hyperopt.STATUS_OK,
                    'model': model}

        trials = hyperopt.Trials()
        hyperopt.fmin(
            eval_params,
            space,
            algo=hyperopt.tpe.suggest,
            max_evals=self.max_evals,
            trials=trials)
        best_trial = trials.best_trial
        self.model = best_trial['result']['model']
        print best_trial['misc']['vals']

        # -- In principle some warm-start refinement of the best model using
        #    the whole data set should be an improvement, but I can't
        #    currently avoid seeing an error from skdata about
        #    a non-contiguous array when I try to do this:
        #
        # assert self.model.warm_start == False
        # self.model.warm_start = True
        # self.model.fit(X, y)
        #
        # -- Also, the Perceptron algorithm can be a bit finicky about initial
        #    conditions. For example, I tried fitting the whole data from
        #    scratch using the best kwargs from the search, but at least on
        #    the Iris data this degraded performance from 0.14 -> 0.27 average
        #    error across folds.  It's safer to just forget about that little
        #    bit of extra data, and go with a model that has proven itself
        #    (as-is) on validation data.

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)



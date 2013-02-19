"""
This file provides AutoPerceptron: an sklearn-style learning algorithm that
optimizes the hyper-parameters of sklearn.linear_model.Perceptron

"""

import numpy as np
import sklearn.linear_model
import hyperopt


def search_space(
    make_name,
    n_iter=5,
    shuffle='opt',
    verbose=0,
    alpha0=0.0001,
    alpha0std=2,
    lr0=1.0,
    lr0std=1.0,
    n_jobs=1,
    class_weight='auto',
    warm_start=False,
    ):
    """
    Build a hyperopt search space for sklearn's perceptron.

    See sklearn's documentation for the meaning of some of these variables.

    Parameters
    ----------

    alpha0 and alpha0std parameterize a lognormal distribution over the
    regularization strength.

    lr0 and lr0std to parameterize a lognormal distribution over the learning
    rate eta0.

    shuffle - int or 'opt'
        'opt' means optimize the strategy for data set shuffling;
        0 means do not shuffle the data set;
        any other integer means shuffle using `shuffle` to seed the random
        shuffling process.

    """
    hp = hyperopt.hp

    l2_reg = dict(
        penalty='l2',
        alpha=hp.lognormal(make_name('l2_reg'), np.log(alpha0), 2))

    l1_reg = dict(
        penalty='l1',
        alpha=hp.lognormal(make_name('l1_reg'), np.log(alpha0), 2))

    el_reg = dict(
        penalty='elasticnet',
        alpha=hp.lognormal(make_name('le_reg'), np.log(alpha0), 2))

    regularization = hp.choice(make_name('regularization'), 
                               [dict(penalty=None, alpha=0.0001),
                                l2_reg,
                                l1_reg,
                                el_reg])

    if shuffle == 'opt':
        shuffle_seed = hp.choice(make_name('shuffle_seed'), range(100, 105))
        hp_shuffle = hp.choice(
            make_name('shuffle'), [(False, None), (True, shuffle_seed)])
    elif shuffle:
        hp_shuffle = hp.choice(
            make_name('shuffle'), [(True, shuffle)])
    else:
        hp_shuffle = hp.choice(
            make_name('shuffle'), [(False, None)])

    rval = dict(
        penalty=regularization['penalty'],
        alpha=regularization['alpha'],
        fit_intercept=hp.choice(make_name('fit_intercept'), [False, True]),
        n_iter=n_iter,
        shuffle=hp_shuffle[0],
        random_state=hp_shuffle[1],
        verbose=verbose,
        n_jobs=n_jobs,
        eta0=hp.lognormal(make_name('eta0'), np.log(lr0), lr0std),
        class_weight=class_weight,
        warm_start=warm_start,
        )

    return rval


class AutoPerceptron(object):
    def __init__(self,
                 n_valid=0.2,
                 algo=hyperopt.tpe.suggest,
                 max_evals=100,
                 search_space=search_space):
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
            model = sklearn.linear_model.Perceptron(**kw)
            try:
                model.fit(X_fit, y_fit)
            except ValueError, e:
                if 'overflow' in str(e):
                    # This can happen once in a while, maybe due to large
                    # learning rate? Or large regularization?
                    return err_rval
                raise

            y_hat = model.predict(X_val)
            val_erate = np.mean(y_hat != y_val)
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



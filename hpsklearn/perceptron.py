"""
This file provides AutoPerceptron: an sklearn-style learning algorithm that
optimizes the hyper-parameters of sklearn.linear_model.Perceptron

"""

import numpy as np
import hyperopt

from sklearn.linear_model import Perceptron

from hpcv import HyperoptMetaEstimator
from hpcv import classification_loss


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


class AutoPerceptron(HyperoptMetaEstimator):
    def __init__(self, **kwargs):
        HyperoptMetaEstimator.__init__(
            self,
            estimator=Perceptron,
            search_space=search_space,
            valid_loss_fn=classification_loss,
            **kwargs
            )


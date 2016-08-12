from __future__ import print_function
# import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from hyperopt import tpe
import hpsklearn
import sys

def test_demo_iris():

    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=.25, random_state=1)

    estimator = hpsklearn.HyperoptEstimator(
        preprocessing=hpsklearn.components.any_preprocessing('pp'),
        classifier=hpsklearn.components.any_classifier('clf'),
        algo=tpe.suggest,
        trial_timeout=15.0,  # seconds
        max_evals=10,
        seed=1
    )

    # /BEGIN `Demo version of estimator.fit()`
    print('', file=sys.stderr)
    print('====Demo classification on Iris dataset====', file=sys.stderr)

    iterator = estimator.fit_iter(X_train, y_train)
    next(iterator)

    n_trial = 0
    while len(estimator.trials.trials) < estimator.max_evals:
        iterator.send(1)  # -- try one more model
        n_trial += 1
        print('Trial', n_trial, 'loss:', estimator.trials.losses()[-1], 
              file=sys.stderr)
        # hpsklearn.demo_support.scatter_error_vs_time(estimator)
        # hpsklearn.demo_support.bar_classifier_choice(estimator)

    estimator.retrain_best_model_on_full_data(X_train, y_train)

    # /END Demo version of `estimator.fit()`

    print('Test accuracy:', estimator.score(X_test, y_test), file=sys.stderr)
    print('====End of demo====', file=sys.stderr)


def test_demo_boston():

    boston = datasets.load_boston()
    X_train, X_test, y_train, y_test = train_test_split(
        boston.data, boston.target, test_size=.25, random_state=1)

    estimator = hpsklearn.HyperoptEstimator(
        preprocessing=hpsklearn.components.any_preprocessing('pp'),
        regressor=hpsklearn.components.any_regressor('reg'),
        algo=tpe.suggest,
        trial_timeout=15.0,  # seconds
        max_evals=10,
        seed=1
    )

    # /BEGIN `Demo version of estimator.fit()`
    print('', file=sys.stderr)
    print('====Demo regression on Boston dataset====', file=sys.stderr)


    iterator = estimator.fit_iter(X_train, y_train)
    next(iterator)

    n_trial = 0
    while len(estimator.trials.trials) < estimator.max_evals:
        iterator.send(1)  # -- try one more model
        n_trial += 1
        print('Trial', n_trial, 'loss:', estimator.trials.losses()[-1], 
              file=sys.stderr)
        # hpsklearn.demo_support.scatter_error_vs_time(estimator)
        # hpsklearn.demo_support.bar_classifier_choice(estimator)

    estimator.retrain_best_model_on_full_data(X_train, y_train)

    # /END Demo version of `estimator.fit()`

    print('Test R2:', estimator.score(X_test, y_test), file=sys.stderr)
    print('====End of demo====', file=sys.stderr)


# -- flake8 eof

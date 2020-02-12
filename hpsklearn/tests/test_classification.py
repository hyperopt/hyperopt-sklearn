try:
    import unittest2 as unittest
except:
    import unittest

import numpy as np
from hyperopt import rand, tpe
from hpsklearn.estimator import hyperopt_estimator
from hpsklearn import components


class TestClassification(unittest.TestCase):

    def setUp(self):
        np.random.seed(123)
        self.X_train = np.random.randn(1000, 2)
        self.Y_train = (self.X_train[:, 0] > 0).astype('int')
        self.Y_train_multilabel = (self.X_train[:, :] > 0).astype('int')
        self.X_test = np.random.randn(1000, 2)
        self.Y_test = (self.X_test[:, 0] > 0).astype('int')
        self.Y_test_multilabel = (self.X_test[:, :] > 0).astype('int')

    def test_multinomial_nb(self):
        model = hyperopt_estimator(
            classifier=components.multinomial_nb('classifier'),
            preprocessing=[],
            algo=rand.suggest,
            trial_timeout=5.0,
            max_evals=5,
        )
        
        # Inputs for MultinomialNB must be non-negative
        model.fit(np.abs(self.X_train), self.Y_train)
        model.score(np.abs(self.X_test), self.Y_test)

def create_function(clf_fn):
    def test_classifier(self):
        model = hyperopt_estimator(
            classifier=clf_fn('classifier'),
            preprocessing=[],
            algo=rand.suggest,
            trial_timeout=5.0,
            max_evals=5,
        )
        model.fit(self.X_train, self.Y_train)
        model.score(self.X_test, self.Y_test)

    test_classifier.__name__ = 'test_{0}'.format(clf_fn.__name__)
    return test_classifier

def create_multilabel_function(clf_fn):
    def test_classifier(self):
        model = hyperopt_estimator(
            classifier=clf_fn('classifier'),
            preprocessing=[],
            algo=rand.suggest,
            trial_timeout=5.0,
            max_evals=5,
        )
        model.fit(self.X_train, self.Y_train_multilabel)
        model.score(self.X_test, self.Y_test_multilabel)

    test_classifier.__name__ = 'test_{0}'.format(clf_fn.__name__)
    return test_classifier

# List of classifiers to test
classifiers = [
    components.svc,
    components.svc_linear,
    components.svc_rbf,
    components.svc_poly,
    components.svc_sigmoid,
    components.liblinear_svc,
    components.knn,
    components.ada_boost,
    components.gradient_boosting,
    components.random_forest,
    components.extra_trees,
    components.decision_tree,
    components.sgd,
    #components.multinomial_nb,  # special case to ensure non-negative inputs
    components.gaussian_nb,
    components.passive_aggressive,
    components.linear_discriminant_analysis,
    components.one_vs_one,
    components.output_code,
]

# Special case for classifiers supporting multiple labels
multilabel_classifiers = [
    components.one_vs_rest,
]

# Create unique methods with test_ prefix so that nose can see them
for clf in classifiers:
    setattr(
        TestClassification,
        'test_{0}'.format(clf.__name__),
        create_function(clf)
    )

for clf in multilabel_classifiers:
    setattr(
        TestClassification,
        'test_{0}'.format(clf.__name__),
        create_multilabel_function(clf)
    )

# Only test the xgboost classifier if the optional dependency is installed
try:
    import xgboost
except ImportError:
    xgboost = None

if xgboost is not None:
    setattr(
        TestClassification,
        'test_xgboost_classification',
        create_function(components.xgboost_classification)
    )

# Only test the lightgbm classifier if the optional dependency is installed
try:
    import lightgbm
except ImportError:
    lightgbm = None

if lightgbm is not None:
    setattr(
        TestClassification,
        'test_lightgbm_classification',
        create_function(components.lightgbm_classification)
    )

if __name__ == '__main__':
    unittest.main()

# -- flake8 eof

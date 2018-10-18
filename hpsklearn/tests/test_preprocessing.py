try:
    import unittest2 as unittest
except:
    import unittest

import numpy as np
from hyperopt import rand, tpe, hp
from hpsklearn.estimator import hyperopt_estimator
from hpsklearn import components
from hyperopt.pyll import scope


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        np.random.seed(123)
        self.X_train = np.random.randn(1000, 2)
        self.Y_train = (self.X_train[:, 0] > 0).astype('int')
        self.X_test = np.random.randn(1000, 2)
        self.Y_test = (self.X_test[:, 0] > 0).astype('int')
    
    def test_one_hot_encoder(self):
        # requires a classifier that can handle sparse data
        model = hyperopt_estimator(
            classifier=components.multinomial_nb('classifier'),
            preprocessing=[components.one_hot_encoder('preprocessing')],
            algo=rand.suggest,
            trial_timeout=5.0,
            max_evals=5,
        )
        
        # Inputs for one_hot_encoder must be non-negative integers
        model.fit(np.abs(np.round(self.X_test).astype(np.int)), self.Y_test)
        model.score(np.abs(np.round(self.X_test).astype(np.int)), self.Y_test)
    
    def test_tfidf(self):
        # requires a classifier that can handle sparse data
        model = hyperopt_estimator(
            classifier=components.multinomial_nb('classifier'),
            preprocessing=[components.tfidf('preprocessing')],
            algo=rand.suggest,
            trial_timeout=5.0,
            max_evals=5,
        )

        X = np.array([
            'This is the first document.',
            'This document is the second document.',
            'And this is the third one.',
            'Is this the first document?',
        ])

        Y = np.array([0, 1, 2, 0])
        
        model.fit(X, Y)
        model.score(X, Y)

    def test_gaussian_random_projection(self):
        # restrict n_components to be less than or equal to data dimension
        # to prevent sklearn warnings from printing during tests
        n_components = scope.int(hp.quniform(
            'preprocessing.n_components', low=1, high=8, q=1
        ))
        model = hyperopt_estimator(
            classifier=components.gaussian_nb('classifier'),
            preprocessing=[
                components.gaussian_random_projection(
                    'preprocessing',
                    n_components=n_components,
                )
            ],
            algo=rand.suggest,
            trial_timeout=5.0,
            max_evals=5,
        )

        X_train = np.random.randn(1000, 8)
        Y_train = (self.X_train[:, 0] > 0).astype('int')
        X_test = np.random.randn(1000, 8)
        Y_test = (self.X_test[:, 0] > 0).astype('int')

        model.fit(X_train, Y_train)
        model.score(X_test, Y_test)

    def test_sparse_random_projection(self):
        # restrict n_components to be less than or equal to data dimension
        # to prevent sklearn warnings from printing during tests
        n_components = scope.int(hp.quniform(
            'preprocessing.n_components', low=1, high=8, q=1
        ))
        model = hyperopt_estimator(
            classifier=components.gaussian_nb('classifier'),
            preprocessing=[
                components.sparse_random_projection(
                    'preprocessing',
                    n_components=n_components,
                )
            ],
            algo=rand.suggest,
            trial_timeout=5.0,
            max_evals=5,
        )

        X_train = np.random.randn(1000, 8)
        Y_train = (self.X_train[:, 0] > 0).astype('int')
        X_test = np.random.randn(1000, 8)
        Y_test = (self.X_test[:, 0] > 0).astype('int')

        model.fit(X_train, Y_train)
        model.score(X_test, Y_test)

def create_function(pre_fn):
    def test_preprocessing(self):
        model = hyperopt_estimator(
            classifier=components.gaussian_nb('classifier'),
            preprocessing=[pre_fn('preprocessing')],
            algo=rand.suggest,
            trial_timeout=5.0,
            max_evals=5,
        )
        model.fit(self.X_train, self.Y_train)
        model.score(self.X_test, self.Y_test)

    test_preprocessing.__name__ = 'test_{0}'.format(pre_fn.__name__)
    return test_preprocessing


# List of preprocessors to test
preprocessors = [
    components.pca,
    #components.one_hot_encoder,  # handled separately 
    components.standard_scaler,
    components.min_max_scaler,
    components.normalizer,
    #components.ts_lagselector,  # handled in test_ts.py
    #components.tfidf,  # handled separately
    #components.sparse_random_projection,  # handled separately
    #components.gaussian_random_projection,  # handled separately
]


# Create unique methods with test_ prefix so that nose can see them
for pre in preprocessors:
    setattr(
        TestPreprocessing,
        'test_{0}'.format(pre.__name__),
        create_function(pre)
    )

if __name__ == '__main__':
    unittest.main()

# -- flake8 eof

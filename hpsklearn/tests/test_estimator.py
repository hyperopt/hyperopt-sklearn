
try:
    import unittest2 as unittest
except:
    import unittest

import numpy as np
from hyperopt import rand, tpe
from hpsklearn.estimator import hyperopt_estimator
from hpsklearn import components


class TestIter(unittest.TestCase):

    def setUp(self):
        np.random.seed(123)
        self.X = np.random.randn(1000, 2)
        self.Y = (self.X[:, 0] > 0).astype('int')

    def test_fit_iter_basic(self):
        model = hyperopt_estimator(
            classifier=components.any_classifier('classifier'), 
            verbose=1, trial_timeout=5.0)
        for ii, trials in enumerate(model.fit_iter(self.X, self.Y)):
            assert trials is model.trials
            assert len(trials.trials) == ii
            if ii == 10:
                break

    def test_fit(self):
        model = hyperopt_estimator(
            classifier=components.any_classifier('classifier'), 
            verbose=1, max_evals=5, trial_timeout=5.0)
        model.fit(self.X, self.Y)
        assert len(model.trials.trials) == 5

    def test_fit_biginc(self):
        model = hyperopt_estimator(
            classifier=components.any_classifier('classifier'),
            verbose=1, max_evals=5, trial_timeout=5.0, fit_increment=20)
        model.fit(self.X, self.Y)
        # -- make sure we only get 5 even with big fit_increment
        assert len(model.trials.trials) == 5

    def test_warm_start(self):
        model = hyperopt_estimator(
            classifier=components.any_classifier('classifier'), 
            verbose=1, max_evals=5, trial_timeout=5.0)
        params = model.get_params()
        assert params['algo'] == rand.suggest
        assert params['max_evals'] == 5
        model.fit(self.X, self.Y, warm_start=False)
        assert len(model.trials.trials) == 5
        model.set_params(algo=tpe.suggest, max_evals=10)
        params = model.get_params()
        assert params['algo'] == tpe.suggest
        assert params['max_evals'] == 10
        model.fit(self.X, self.Y, warm_start=True)
        assert len(model.trials.trials) == 15  # 5 + 10 = 15.

def test_sparse_input():
    """
    Ensure the estimator can handle sparse X matrices.
    """

    import scipy.sparse as ss

    # Generate some random sparse data
    nrows,ncols,nnz = 100,50,10
    ntrue = nrows // 2
    D,C,R = [],[],[]
    for r in range(nrows):
        feats = np.random.choice(range(ncols), size=nnz, replace=False)
        D.extend([1]*nnz)
        C.extend(feats)
        R.extend([r]*nnz)
    X = ss.csr_matrix( (D,(R,C)), shape=(nrows, ncols))
    y = np.zeros( nrows )
    y[:ntrue] = 1


    # Try to fit an SGD model
    cls = hyperopt_estimator(
        classifier=components.sgd('sgd', loss='log'),
        preprocessing=[],
    )
    cls.fit(X,y)

def test_continuous_loss_fn():
    """
    Demonstrate using a custom loss function with the continuous_loss_fn
    option.
    """

    from sklearn.metrics import log_loss

    # Generate some random data
    X = np.hstack([
        np.vstack([
            np.random.normal(0,1,size=(1000,10)),
            np.random.normal(1,1,size=(1000,10)),
        ]),
        np.random.normal(0,1,size=(2000,10)),
    ])
    y = np.zeros(2000)
    y[:1000] = 1

    def loss_function(targ, pred):
        # hyperopt_estimator flattens the prediction when saving it.  This also
        # affects multilabel classification.
        pred = pred.reshape( (-1, 2) )
        return log_loss(targ, pred[:,1])

    # Try to fit an SGD model using log_loss as the loss function
    cls = hyperopt_estimator(
        classifier=components.sgd('sgd', loss='log'),
        preprocessing=[],
        loss_fn = loss_function,
        continuous_loss_fn=True,
    )
    cls.fit(X,y,cv_shuffle=True)


# class TestSpace(unittest.TestCase):

#     def setUp(self):
#         np.random.seed(123)
#         self.X = np.random.randn(1000, 2)
#         self.Y = (self.X[:, 0] > 0).astype('int')

#     def test_smoke(self):
#         # -- verify the space argument is accepted and runs
#         space = components.generic_space()
#         model = hyperopt_estimator(
#             verbose=1, max_evals=10, trial_timeout=5, space=space)
#         model.fit(self.X, self.Y)

# -- flake8 eof

def test_crossvalidation():
    """
    Demonstrate performing a k-fold CV using the fit() method.
    """
    # Generate some random data
    X = np.hstack([
        np.vstack([
            np.random.normal(0,1,size=(1000,10)),
            np.random.normal(1,1,size=(1000,10)),
        ]),
        np.random.normal(0,1,size=(2000,10)),
    ])
    y = np.zeros(2000)
    y[:1000] = 1

    # Try to fit a model
    cls = hyperopt_estimator(
        classifier=components.sgd('sgd', loss='log'),
        preprocessing=[],
    )
    cls.fit(X,y,cv_shuffle=True, n_folds=5)

import unittest

import numpy as np
from hyperopt import rand, tpe
from hyperopt.pyll import as_apply
from hpsklearn import HyperoptEstimator
from hpsklearn import sgd_classifier, any_classifier
from tests.utils import RetryOnTrialsException


class TestIter(unittest.TestCase):
    """
    Class for iteration testing
    """

    def setUp(self):
        """
        Setup of randomly generated data
        """
        np.random.seed(123)
        self.X = np.random.randn(1000, 2)
        self.Y = (self.X[:, 0] > 0).astype("int")

    @RetryOnTrialsException
    def test_no_params(self):
        """
        Test fitting without any parameters
         tests presence of circular imports and variable assignment
        """
        model = HyperoptEstimator(max_evals=5, trial_timeout=5.0)
        params = model.get_params()
        assert params["classifier"] is None
        assert params["regressor"] is None
        assert params["preprocessing"] is None
        assert params["space"] is None
        model.fit(self.X, self.Y)
        params = model.get_params()
        assert params["classifier"] is not None
        assert params["regressor"] is None
        assert params["preprocessing"] is not None
        assert params["space"] is not None

    @RetryOnTrialsException
    def test_fit_iter_basic(self):
        """
        Test fitting with basic iteration
        """
        model = HyperoptEstimator(
            classifier=any_classifier("classifier"),
            verbose=1, trial_timeout=5.0)
        for ii, trials in enumerate(model.fit_iter(self.X, self.Y)):
            assert trials is model.trials
            assert len(trials.trials) == ii
            if ii == 10:
                break

    @RetryOnTrialsException
    def test_fit(self):
        """
        Test fitting
        """
        model = HyperoptEstimator(
            classifier=any_classifier("classifier"),
            verbose=1, max_evals=5, trial_timeout=5.0)
        model.fit(self.X, self.Y)
        assert len(model.trials.trials) == 5

    @RetryOnTrialsException
    def test_fit_biginc(self):
        """
        Test fitting with big fit increment
        """
        model = HyperoptEstimator(
            classifier=any_classifier("classifier"),
            verbose=1, max_evals=5, trial_timeout=5.0, fit_increment=20)
        model.fit(self.X, self.Y)
        # -- make sure we only get 5 even with big fit_increment
        assert len(model.trials.trials) == 5

    @RetryOnTrialsException
    def test_warm_start(self):
        """
        Test fitting with warm start
        """
        model = HyperoptEstimator(
            classifier=any_classifier("classifier"),
            verbose=1, max_evals=5, trial_timeout=5.0)
        params = model.get_params()
        assert params["algo"] == rand.suggest
        assert params["max_evals"] == 5
        assert model.algo == rand.suggest
        assert model.max_evals == 5
        model.fit(self.X, self.Y, warm_start=False)
        assert len(model.trials.trials) == 5
        model.set_params(**{"algo": tpe.suggest, "max_evals": 10})
        params = model.get_params()
        assert params["algo"] == tpe.suggest
        assert params["max_evals"] == 10
        model.fit(self.X, self.Y, warm_start=True)
        assert len(model.trials.trials) == 15  # 5 + 10 = 15.


class TestSparseInput(unittest.TestCase):
    """
    Class for testing estimator with sparse input
    """

    def setUp(self):
        """
        Setup the random seed
        """
        np.random.seed(123)

    @RetryOnTrialsException
    def test_sparse_input(self):
        """
        Ensure the estimator can handle sparse X matrices.
        """
        import scipy.sparse as ss

        # Generate some random sparse data
        nrows, ncols, nnz = 100, 50, 10
        ntrue = nrows // 2
        D, C, R = [], [], []
        for r in range(nrows):
            feats = np.random.choice(range(ncols), size=nnz, replace=False)
            D.extend([1] * nnz)
            C.extend(feats)
            R.extend([r] * nnz)
        X = ss.csr_matrix((D, (R, C)), shape=(nrows, ncols))
        y = np.zeros(nrows)
        y[:ntrue] = 1

        # Try to fit an SGD model
        cls = HyperoptEstimator(classifier=sgd_classifier("sgd", loss="log_loss"),
                                preprocessing=[])
        cls.fit(X, y)


class TestContinuousLossFn(unittest.TestCase):
    """
    Class for testing estimator with continuous loss function
    """

    def setUp(self):
        """
        Setup the random seed
        """
        np.random.seed(123)

    @staticmethod
    def loss_function(targ, pred):
        """
        Custom loss function for testing
         separate function for multiprocess pickling
        """
        from sklearn.metrics import log_loss

        # hyperopt_estimator flattens the prediction when saving it.
        # This also affects multilabel classification.
        pred = pred.reshape((-1, 2))
        return log_loss(targ, pred[:, 1])

    @RetryOnTrialsException
    def test_continuous_loss_fn(self):
        """
        Demonstrate using a custom loss function with the continuous_loss_fn
        option.
        """
        # Generate some random data
        X = np.hstack([
            np.vstack([
                np.random.normal(0, 1, size=(1000, 10)),
                np.random.normal(1, 1, size=(1000, 10)),
            ]),
            np.random.normal(0, 1, size=(2000, 10)),
        ])
        y = np.zeros(2000)
        y[:1000] = 1

        # Try to fit an SGD model using log_loss as the loss function
        cls = HyperoptEstimator(
            classifier=sgd_classifier("sgd", loss="log_loss"),
            preprocessing=[],
            loss_fn=TestContinuousLossFn.loss_function,
            continuous_loss_fn=True,
        )
        cls.fit(X, y, cv_shuffle=True)


class TestSpace(unittest.TestCase):
    """
    Class for testing estimator with custom space
    """
    def setUp(self):
        """
        Setup of randomly generated data
        """
        np.random.seed(123)
        self.X = np.random.randn(1000, 2)
        self.Y = (self.X[:, 0] > 0).astype("int")

    @RetryOnTrialsException
    def test_smoke(self):
        """
        Test that the estimator can be created with a custom space
        """
        # -- verify the space argument is accepted and runs
        space = as_apply({"classifier": sgd_classifier("sgd", loss="log_loss"),
                          "regressor": None,
                          "preprocessing": None,
                          "ex_preprocs": []})
        model = HyperoptEstimator(verbose=1, max_evals=10, trial_timeout=5, space=space)
        model.fit(self.X, self.Y)


class TestCrossValidation(unittest.TestCase):
    """
    Class for testing estimator with cross validation
    """

    def setUp(self):
        """
        Setup the random seed
        """
        np.random.seed(123)

    @RetryOnTrialsException
    def test_crossvalidation(self):
        """
        Demonstrate performing a k-fold CV using the fit() method.
        """
        # Generate some random data
        X = np.hstack([
            np.vstack([
                np.random.normal(0, 1, size=(1000, 10)),
                np.random.normal(1, 1, size=(1000, 10)),
            ]),
            np.random.normal(0, 1, size=(2000, 10)),
        ])
        y = np.zeros(2000)
        y[:1000] = 1

        # Try to fit a model
        cls = HyperoptEstimator(classifier=sgd_classifier("sgd", loss="log_loss"), preprocessing=[])
        cls.fit(X, y, cv_shuffle=True, n_folds=5)


class TestGroupCrossValidation(unittest.TestCase):
    """
    Class for testing estimator with group cross validation
    """

    def setUp(self):
        """
        Setup the random seed
        """
        np.random.seed(123)

    @RetryOnTrialsException
    def test_crossvalidation(self):
        """
        Demonstrate performing a group k-fold CV using the fit() method.
        """
        # Generate some random data
        X = np.hstack([
            np.vstack([
                np.random.normal(0, 1, size=(1000, 10)),
                np.random.normal(1, 1, size=(1000, 10)),
            ]),
            np.random.normal(0, 1, size=(2000, 10)),
        ])
        y = np.zeros(2000)
        y[:1000] = 1

        # Try to fit a model
        cls = HyperoptEstimator(classifier=sgd_classifier("sgd", loss="log_loss"), preprocessing=[])
        cls.fit(X, y, cv_shuffle=True, n_folds=5,
                kfolds_group=np.array([0]*500 + [1]*500 + [2]*500 + [3]*500))  # noqa: E226


if __name__ == '__main__':
    unittest.main()

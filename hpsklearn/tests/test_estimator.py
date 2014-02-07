
try:
    import unittest2 as unittest
except:
    import unittest

import numpy as np
from hpsklearn.estimator import hyperopt_estimator

class TestIter(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.X = np.random.randn(1000, 2)
        self.Y = (self.X[:, 0] > 0).astype('int')

    def test_fit_iter_basic(self):
        model = hyperopt_estimator(verbose=1, trial_timeout=5.0 )
        for ii, trials in enumerate(model.fit_iter(self.X, self.Y)):
            assert trials is model.trials
            assert len(trials.trials) == ii
            if ii == 10:
                break

    def test_fit(self):
        model = hyperopt_estimator(verbose=1, max_evals=5, trial_timeout=5.0)
        model.fit(self.X, self.Y)
        assert len(model.trials.trials) == 5

    def test_fit_biginc(self):
        model = hyperopt_estimator(verbose=1, max_evals=5, trial_timeout=5.0,
                                   fit_increment=20)
        model.fit(self.X, self.Y)
        # -- make sure we only get 5 even with big fit_increment
        assert len(model.trials.trials) == 5


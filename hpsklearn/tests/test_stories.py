"""
Unit tests in the form of user stories.

These unit tests implicitly assert that their syntax is valid,
and that programs of the given form can run. They do not
test the correctness of the result.

"""

from functools import partial
try:
    import unittest2 as unittest
except:
    import unittest


from hyperopt import hp
from hyperopt import tpe
from hyperopt.pyll import scope
from hpsklearn import components as hpc

import skdata.iris.view
from skdata.base import SklearnClassifier
from hpsklearn.estimator import hyperopt_estimator


class SkdataInterface(unittest.TestCase):
    def setUp(self):
        self.view = skdata.iris.view.KfoldClassification(4)

    def test_search_all(self):
        """
        As a ML researcher, I want a quick way to do model selection
        implicitly, in order to get a baseline accuracy score for a new data
        set.

        """
        algo = SklearnClassifier(
            partial(hyperopt_estimator,
                    trial_timeout=15.0, # seconds
                    verbose=1,
                    max_evals=10,
                    ))
        mean_test_error = self.view.protocol(algo)
        print 'mean test error:', mean_test_error

    def test_pca_svm(self):
        """
        As a ML researcher, I want to evaluate a certain parly-defined model
        class, in order to do model-family comparisons.

        For example, PCA followed by linear SVM.

        """
        algo = SklearnClassifier(
            partial(
                hyperopt_estimator,
                preprocessing=[hpc.pca('pca')],
                classifier=hpc.svc_linear('classif'),
                max_evals=10))
        mean_test_error = self.view.protocol(algo)
        print 'mean test error:', mean_test_error

    def test_preproc(self):
        """
        As a domain expert, I have a particular pre-processing that I believe
        reveals important patterns in my data.  I would like to know how good
        a classifier can be built on top of my preprocessing algorithm.
        """

        # -- for testing purpose, suppose that the RBM is our "domain-specific
        #    pre-processing"

        algo = SklearnClassifier(
            partial(
                hyperopt_estimator,
                preprocessing=hp.choice('pp',
                    [
                        # -- VQ (alone)
                        [
                            hpc.colkmeans('vq0',
                                n_init=1),
                        ],
                        # -- VQ -> RBM
                        [
                            hpc.colkmeans('vq1',
                                n_clusters=scope.int(
                                    hp.quniform(
                                        'vq1.n_clusters', 1, 5, q=1)),
                                n_init=1),
                            hpc.rbm(name='rbm:alone',
                                verbose=0)
                        ],
                        # -- VQ -> RBM -> PCA
                        [
                            hpc.colkmeans('vq2',
                                n_clusters=scope.int(
                                    hp.quniform(
                                        'vq2.n_clusters', 1, 5, q=1)),
                                n_init=1),
                            hpc.rbm(name='rbm:pre-pca',
                                verbose=0),
                            hpc.pca('pca')
                        ],
                    ]),
                classifier=hpc.any_classifier('classif'),
                algo=tpe.suggest,
                max_evals=10,
                ))
        mean_test_error = self.view.protocol(algo)
        print 'mean test error:', mean_test_error


# -- TODO: develop tests with pure sklearn stories
if __name__ == '__main__':
    unittest.main()


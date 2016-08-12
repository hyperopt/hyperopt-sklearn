"""
Unit tests in the form of user stories.

These unit tests implicitly assert that their syntax is valid,
and that programs of the given form can run. They do not
test the correctness of the result.

"""
from __future__ import print_function
import sys
from functools import partial
try:
    import unittest2 as unittest
except:
    import unittest
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from hyperopt import hp
from hyperopt import tpe
from hyperopt.pyll import scope
from hpsklearn import components as hpc
try:
    import skdata.iris.view as iris_view
except ImportError:
    import skdata.iris.views as iris_view
try:
    from skdata.base import SklearnClassifier as LearningAlgo
except ImportError:
    from skdata.base import LearningAlgo as LearningAlgo
from hpsklearn.estimator import hyperopt_estimator


class SkdataInterface(unittest.TestCase):

    def setUp(self):
        self.view = iris_view.KfoldClassification(4)

    def test_search_all(self):
        """
        As a ML researcher, I want a quick way to do model selection
        implicitly, in order to get a baseline accuracy score for a new data
        set.

        """
        algo = LearningAlgo(
            partial(hyperopt_estimator,
                    classifier=hpc.any_classifier('classifier'),
                    # trial_timeout=15.0,  # seconds
                    verbose=1,
                    max_evals=10,
                    ))
        mean_test_error = self.view.protocol(algo)
        print('\n====Iris: any preprocessing + any classifier====', 
              file=sys.stderr)
        print('mean test error:', mean_test_error, file=sys.stderr)
        print('====End optimization====', file=sys.stderr)

    def test_pca_svm(self):
        """
        As a ML researcher, I want to evaluate a certain parly-defined model
        class, in order to do model-family comparisons.

        For example, PCA followed by linear SVM.

        """
        algo = LearningAlgo(
            partial(
                hyperopt_estimator,
                preprocessing=[hpc.pca('pca')],
                classifier=hpc.svc_linear('classif'),
                # trial_timeout=30.0,  # seconds
                verbose=1,
                max_evals=10))
        mean_test_error = self.view.protocol(algo)
        print('\n====Iris: PCA + SVM====', file=sys.stderr)
        print('mean test error:', mean_test_error, file=sys.stderr)
        print('====End optimization====', file=sys.stderr)

    def test_preproc(self):
        """
        As a domain expert, I have a particular pre-processing that I believe
        reveals important patterns in my data.  I would like to know how good
        a classifier can be built on top of my preprocessing algorithm.
        """

        # -- for testing purpose, suppose that the RBM is our "domain-specific
        #    pre-processing"

        algo = LearningAlgo(
            partial(
                hyperopt_estimator,
                preprocessing=hp.choice(
                    'pp',
                    [
                        # -- VQ (alone)
                        [
                            hpc.colkmeans(
                                'vq0',
                                n_clusters=scope.int(
                                    hp.quniform(
                                        'vq0.n_clusters', 1.5, 5.5, q=1)),
                                n_init=1,
                                max_iter=100),
                        ],
                        # -- VQ -> RBM
                        [
                            hpc.colkmeans(
                                'vq1',
                                n_clusters=scope.int(
                                    hp.quniform(
                                        'vq1.n_clusters', 1.5, 5.5, q=1)),
                                n_init=1,
                                max_iter=100),
                            hpc.rbm(name='rbm:alone',
                                    n_components=scope.int(
                                        hp.qloguniform(
                                            'rbm1.n_components',
                                            np.log(4.5), np.log(20.5), 1)),
                                    n_iter=100,
                                    verbose=0)
                        ],
                        # -- VQ -> RBM -> PCA
                        [
                            hpc.colkmeans(
                                'vq2',
                                n_clusters=scope.int(
                                    hp.quniform(
                                        'vq2.n_clusters', 1.5, 5.5, q=1)),
                                n_init=1,
                                max_iter=100),
                            hpc.rbm(name='rbm:pre-pca',
                                    n_components=scope.int(
                                        hp.qloguniform(
                                            'rbm2.n_components',
                                            np.log(4.5), np.log(20.5), 1)),
                                    n_iter=100,
                                    verbose=0),
                            hpc.pca('pca')
                        ],
                    ]),
                classifier=hpc.any_classifier('classif'),
                algo=tpe.suggest,
                #trial_timeout=5.0,  # seconds
                verbose=1,
                max_evals=10,
            ))
        mean_test_error = self.view.protocol(algo)
        print('\n====Iris: VQ + RBM + PCA + any classifier====', 
              file=sys.stderr)
        print('mean test error:', mean_test_error, file=sys.stderr)
        print('====End optimization====')


# -- TODO: develop tests with pure sklearn stories
if __name__ == '__main__':
    unittest.main()

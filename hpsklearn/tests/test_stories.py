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

import sklearn

from hyperopt import hp
from hpsklearn import components as hpc

import skdata.iris.view
from skdata.base import SklearnClassifier
#from hpsklearn.perceptron import AutoPerceptron
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
        algo = SklearnClassifier(hyperopt_estimator)
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
        my_algo = hpc.rbm

        algo = SklearnClassifier(
            partial(
                hyperopt_estimator,
                preprocessing=hp.choice('pp', [
                    [my_algo(name='alone')],
                    [my_algo(name='pre_pca'), pca('pca')],
                    ]),
                classifier=any_classifier('classif')))
        mean_test_error = self.view.protocol(algo)
        print 'mean test error:', mean_test_error

# -- TODO: develop tests with pure sklearn stories
if __name__ == '__main__':
    unittest.main()

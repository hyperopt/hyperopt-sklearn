from functools import partial
try:
    import unittest2 as unittest
except:
    import unittest

import skdata.larochelle_etal_2007.view
from skdata.base import SklearnClassifier
from hyperopt import tpe

from hpsklearn.small_img import simple_small_image_preprocessing
from hpsklearn.estimator import hyperopt_estimator
from hpsklearn import components as hpc


class TestSmallImages(unittest.TestCase):
    """
    Tests that involve experiments on small rasterized images.
    """

    def setUp(self):
        self.algo = SklearnClassifier(
            partial(
                hyperopt_estimator,
                preprocessing=simple_small_image_preprocessing('pp'),
                classifier=hpc.any_classifier('classif'),
                max_evals=100,
                verbose=1,
                algo=tpe.suggest,
                fit_timeout=5.0, # -- seconds
                ))


    def test_rectangles(self):
        view_module = skdata.larochelle_etal_2007.view
        view = view_module.RectanglesVectorXV()
        mean_test_error = view.protocol(self.algo)
        print 'mean test error:', mean_test_error

    def test_convex(self):
        view_module = skdata.larochelle_etal_2007.view
        view = view_module.ConvexVectorXV()
        mean_test_error = view.protocol(self.algo)
        print 'mean test error:', mean_test_error

if __name__ == '__main__':
    unittest.main()

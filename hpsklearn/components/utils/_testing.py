import unittest

import numpy as np


class StandardClassifierTest(unittest.TestCase):
    """
    Standard class for classification testing
    """
    def setUp(self):
        """
        Setup of randomly generated data
        """
        np.random.seed(123)
        self.X_train = np.random.randn(1000, 2)
        self.Y_train = (self.X_train[:, 0] > 0).astype("int")
        self.X_test = np.random.randn(1000, 2)
        self.Y_test = (self.X_test[:, 0] > 0).astype("int")


class StandardRegressorTest(unittest.TestCase):
    """
    Standard class for regressor testing
    """
    def setUp(self):
        """
        Setup of randomly generated data
        """
        np.random.seed(123)
        self.X_train = np.random.randn(1000, 2)
        self.Y_train = self.X_train[:, 0] * 2
        self.X_test = np.random.randn(1000, 2)
        self.Y_test = self.X_test[:, 0] * 2

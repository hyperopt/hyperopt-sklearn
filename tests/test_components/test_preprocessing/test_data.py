import unittest

from hyperopt import rand

from tests.utils import \
    StandardPreprocessingTest, \
    generate_preprocessor_attributes, \
    TrialsExceptionHandler
from hpsklearn import HyperoptEstimator, \
    binarizer, \
    min_max_scaler, \
    max_abs_scaler, \
    normalizer, \
    robust_scaler, \
    standard_scaler, \
    quantile_transformer, \
    power_transformer, \
    gaussian_nb

import numpy as np


class TestDataPreprocessing(StandardPreprocessingTest):
    """
    Class for _data preprocessing testing
    """
    @TrialsExceptionHandler
    def test_power_transformer(self):
        """
        Instantiate gaussian_nb hyperopt estimator model
         define preprocessor power_transformer
         fit and score model on positive data
        """
        model = HyperoptEstimator(
            classifier=gaussian_nb("classifier"),
            preprocessing=[power_transformer("preprocessing")],
            algo=rand.suggest,
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(np.abs(self.X_train), self.Y_train)
        model.score(np.abs(self.X_test), self.Y_test)

    test_power_transformer.__name__ = f"test_{power_transformer.__name__}"


# List of preprocessors to test
preprocessors = [
    binarizer,
    min_max_scaler,
    max_abs_scaler,
    normalizer,
    robust_scaler,
    standard_scaler,
    quantile_transformer
]


generate_preprocessor_attributes(
    TestClass=TestDataPreprocessing,
    preprocessor_list=preprocessors,
    classifier=gaussian_nb,
)


if __name__ == '__main__':
    unittest.main()

import unittest

from hyperopt import rand

from hpsklearn.tests.utils import \
    StandardPreprocessingTest
from hpsklearn import hyperopt_estimator, \
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
    def test_power_transformer(self):
        """
        Instantiate gaussian_nb hyperopt estimator model
         define preprocessor power_transformer
         fit and score model on positive data
        """
        model = hyperopt_estimator(
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


def create_preprocessing_function(pre_fn):
    """
    Instantiate gaussian_nb hyperopt estimator model
     'pre_fn' regards the preprocessor
     fit and score model
    """
    def test_preprocessor(self):
        model = hyperopt_estimator(
            classifier=gaussian_nb("classifier"),
            preprocessing=[pre_fn("preprocessing")],
            algo=rand.suggest,
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(self.X_train, self.Y_train)
        model.score(self.X_test, self.Y_test)

    test_preprocessor.__name__ = f"test_{pre_fn.__name__}"
    return test_preprocessor


# Create unique _data preprocessing algorithms
#  with test_ prefix so that nose can see them
for pre in preprocessors:
    setattr(
        TestDataPreprocessing,
        f"test_{pre.__name__}",
        create_preprocessing_function(pre)
    )


if __name__ == '__main__':
    unittest.main()

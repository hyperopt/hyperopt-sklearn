import unittest
import numpy as np

from hyperopt import rand

from hpsklearn.tests.utils import \
    StandardPreprocessingTest
from hpsklearn import hyperopt_estimator, \
    min_max_scaler, \
    normalizer, \
    one_hot_encoder, \
    standard_scaler, \
    gaussian_nb, \
    multinomial_nb


class TestDataPreprocessing(StandardPreprocessingTest):
    """
    Class for _data preprocessing testing
    """
    def test_one_hot_encoder(self):
        """
        Instantiate multinomial_nb hyperopt estimator model
         define preprocessor one_hot_encoder
         fit and score model on test set
        """
        model = hyperopt_estimator(
            classifier=multinomial_nb("classifier"),
            preprocessing=[one_hot_encoder("preprocessing")],
            algo=rand.suggest,
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(np.abs(np.round(self.X_test).astype(np.int64)), self.Y_test)
        model.score(np.abs(np.round(self.X_test).astype(np.int64)), self.Y_test)


# List of preprocessors to test
preprocessors = [
    min_max_scaler,
    normalizer,
    standard_scaler
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

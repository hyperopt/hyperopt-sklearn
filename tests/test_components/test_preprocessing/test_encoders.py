import unittest
import numpy as np

from hyperopt import rand

from tests.utils import \
    StandardPreprocessingTest, \
    TrialsExceptionHandler
from hpsklearn import HyperoptEstimator, \
    one_hot_encoder, \
    ordinal_encoder, \
    multinomial_nb


class TestEncodersPreprocessing(StandardPreprocessingTest):
    """
    Class for _encoders preprocessing testing
    """


preprocessors = [
    one_hot_encoder,
    ordinal_encoder,
]


def create_preprocessing_function(pre_fn):
    """
    Instantiate multinomial_nb hyperopt estimator model
     'pre_fn' regards the preprocessor
     fit and score model
    """

    @TrialsExceptionHandler
    def test_preprocessor(self):
        model = HyperoptEstimator(
            classifier=multinomial_nb("classifier"),
            preprocessing=[pre_fn("preprocessing")],
            algo=rand.suggest,
            trial_timeout=5.0,
            max_evals=5,
        )
        model.fit(np.abs(np.round(self.X_test).astype(np.int64)), self.Y_test)
        model.score(np.abs(np.round(self.X_test).astype(np.int64)), self.Y_test)

    test_preprocessor.__name__ = f"test_{pre_fn.__name__}"
    return test_preprocessor


# Create unique _data preprocessing algorithms
#  with test_ prefix so that unittest can see them
for pre in preprocessors:
    setattr(
        TestEncodersPreprocessing,
        f"test_{pre.__name__}",
        create_preprocessing_function(pre)
    )


if __name__ == '__main__':
    unittest.main()

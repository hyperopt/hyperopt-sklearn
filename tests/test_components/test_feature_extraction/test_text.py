import unittest

from hpsklearn import \
    HyperoptEstimator, \
    multinomial_nb, \
    sgd_classifier, \
    tfidf_vectorizer, \
    hashing_vectorizer, \
    count_vectorizer
from tests.utils import TrialsExceptionHandler
from hyperopt import rand

import numpy as np


class TestTfidf(unittest.TestCase):
    """
    Class for TfidfVectorizer testing
    """
    def setUp(self):
        """
        Set up text data
        """
        self.X_test = np.array([
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
        ])

        self.Y_test = np.array([0, 1, 2, 0])

    @TrialsExceptionHandler
    def test_tfidf_vectorizer(self):
        """
        Instantiate multinomial_nb hyperopt_estimator model
         add TfidfVectorizer preprocessor
         fit and score model
        """
        model = HyperoptEstimator(
            classifier=multinomial_nb("classifier"),
            preprocessing=[tfidf_vectorizer("preprocessing")],
            algo=rand.suggest,
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(self.X_test, self.Y_test)
        model.score(self.X_test, self.Y_test)

    @TrialsExceptionHandler
    def test_hashing_vectorizer(self):
        """
        Instantiate sgd_classifier hyperopt_estimator model
         add HashingVectorizer preprocessor
         fit and score model
        """
        model = HyperoptEstimator(
            classifier=sgd_classifier("classifier"),
            preprocessing=[hashing_vectorizer("preprocessing")],
            algo=rand.suggest,
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(self.X_test, self.Y_test)

    @TrialsExceptionHandler
    def test_count_vectorizer(self):
        """
        Instantiate multinomial_nb hyperopt_estimator model
         add CountVectorizer preprocessor
         fit and score model
        """
        model = HyperoptEstimator(
            classifier=multinomial_nb("classifier"),
            preprocessing=[count_vectorizer("preprocessing")],
            algo=rand.suggest,
            trial_timeout=10.0,
            max_evals=5,
        )
        model.fit(self.X_test, self.Y_test)

    test_tfidf_vectorizer.__name__ = f"test_{tfidf_vectorizer.__name__}"
    test_hashing_vectorizer.__name__ = f"test_{hashing_vectorizer.__name__}"
    test_count_vectorizer.__name__ = f"test_{count_vectorizer.__name__}"


if __name__ == "__main__":
    unittest.main()

"""Lag selectors that subset time series predictors

This module defines lag selectors with specified lag sizes for endogenous and 
exogenous predictors, using the same style as the sklearn transformers. They 
can be used in hpsklearn as preprocessors. The module is well suited for time 
series data.

When use a lag size of a positive integer, it is assumed that lag=1, 2, ... 
predictors are located at the 1st, 2nd, ... columns. When use a negative 
integer, the predictors are located at the N-th, (N - 1)th, ... columns.
 
"""
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class LagSelector(BaseEstimator, TransformerMixin):
    """Subset time series features by choosing the most recent lags

    Parameters
    ----------
    lag_size : int, None by default
        If None, use all features. If positive integer, use features by 
        subsetting the X as [:, :lag_size]. If negative integer, use features 
        by subsetting the X as [:, lag_size:]. If 0, discard the features 
        from this dataset.

    Attributes
    ----------
    max_lag_size_ : int
        The largest allowed lag size inferred from input.
    """

    def __init__(self, lag_size=None):
        self.lag_size = lag_size

    def _reset(self):
        """Reset internal data-dependent state of the selector, if necessary.

        __init__ parameters are not touched.
        """
        if hasattr(self, 'max_lag_size_'):
            del self.max_lag_size_

    def fit(self, X, y=None):
        """Infer the maximum lag size.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The input time series data with lagged predictors as features.

        y: Passthrough for ``Pipeline`` compatibility.
        """

        # Reset internal state before fitting
        self._reset()
        self.max_lag_size_ = X.shape[1]

    def transform(self, X, y=None):
        """Perform standardization by centering and scaling

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The input time series data with lagged predictors as features.
        """
        proofed_lag_size = min(self.max_lag_size_, abs(self.lag_size))
        if self.lag_size >= 0:
            return X[:, :proofed_lag_size]
        else:
            return X[:, -proofed_lag_size:]










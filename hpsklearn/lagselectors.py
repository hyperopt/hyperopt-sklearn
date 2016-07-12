"""Lag selectors that extend sklearn regressors and classifiers
This module defines child classes of sklearn learners, which have an option 
to specify lag sizes for endogenous and exogenous predictors. This is well 
suited for time series data.
For any endogenous or exogenous dataset, it is assumed that lag=1 predictors 
are located at the 1st column, lag=2 predictors are located at the 2nd column, 
and so on.
Each class has the following additional arguments to the original sklearn 
arguments in this module:
    en_nlag ([int]): lag size of endogenous predictors.
    ex_nlag ([int or list of int]): 
            Lag size(s) of exogenous predictors. If int, a single lag 
        size for all kinds of exogenous predictors; if a list of int, 
        can specify different lag sizes for each kind of exogenous 
        predictors separately. 
"""
import numpy as np
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

class LagSelector:
    '''Base class for all lag selectors
    '''
    def check_set_nlag(self, en_nlag=None, ex_nlag=None):
        if ex_nlag is not None:
            assert (isinstance(ex_nlag, int) or \
                isinstance(ex_nlag, (list, tuple)))
        self.en_nlag = en_nlag
        self.ex_nlag = ex_nlag

def _check_assemble(X, EX_list, en_nlag, ex_nlag, lags):
    '''Check if the input endogenous and exogenous matrices have 
    correct sizes. If they are incorrect, generate assertion errors. 
    Else, return a matrix with assembled endogenous and exogenous 
    datasets. 
    '''
    if not lags:
        if EX_list is None:
            return X
        else:
            return np.c_[X, np.concatenate(EX_list, axis=1)]

    assert en_nlag <= X.shape[1]
    if EX_list is not None:
        if isinstance(ex_nlag, int):
            ex_nlag = [ex_nlag] * len(EX_list)
        else:
            assert len(ex_nlag) == len(EX_list)
            is_lag_within = [ex_nlag[i] <= EX_list[i].shape[1] \
                for i in range(len(EX_list))]
            assert all(is_lag_within)
        subset_ex_li = [EX_list[i][:, :ex_nlag[i]] \
            for i in range(len(EX_list))]
        return np.c_[X[:, :en_nlag], np.concatenate(subset_ex_li, axis=1)]
    else:
        return X[:, :en_nlag]


class SVRLagSelector(LagSelector, SVR):
    '''SVR child class that uses the numbers of lagged predictors 
    as hyperparameters.
    '''
    def __init__(self, en_nlag=None, ex_nlag=None, 
                 kernel='rbf', degree=3, gamma='auto', coef0=0.0,
                 tol=1e-3, C=1.0, epsilon=0.1, shrinking=True,
                 cache_size=200, verbose=False, max_iter=-1):
        super(SVRLagSelector, self).check_set_nlag(en_nlag, ex_nlag)
        super(SVRLagSelector, self).__init__(
            kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
            tol=tol, C=C, epsilon=epsilon, shrinking=shrinking,
            cache_size=cache_size, verbose=verbose, max_iter=max_iter)

    def fit(self, X, y, EX_list=None, lags=False, sample_weight=None):
        XEX = _check_assemble(X, EX_list, self.en_nlag, self.ex_nlag, lags)
        return super(SVRLagSelector, self).fit(XEX, y, sample_weight)
    
    def predict(self, X, EX_list=None, lags=False):
        XEX = _check_assemble(X, EX_list, self.en_nlag, self.ex_nlag, lags)
        return super(SVRLagSelector, self).predict(XEX)
    
    def score(self, X, y, EX_list=None, lags=False, sample_weight=None):
        XEX = _check_assemble(X, EX_list, self.en_nlag, self.ex_nlag, lags)
        return super(SVRLagSelector, self).score(XEX, y, sample_weight)

class KNRLagSelector(LagSelector, KNeighborsRegressor):
    '''KNN regressor child class that uses the numbers of lagged predictors 
    as hyperparameters.
    '''
    def __init__(self, en_nlag=None, ex_nlag=None, 
                 n_neighbors=5, weights='uniform', algorithm='auto', 
                 leaf_size=30, p=2, metric='minkowski', metric_params=None, 
                 n_jobs=1, **kwargs):
        super(KNRLagSelector, self).check_set_nlag(en_nlag, ex_nlag)
        super(KNRLagSelector, self).__init__(
            n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, 
            leaf_size=leaf_size, p=p, metric=metric, 
            metric_params=metric_params, n_jobs=n_jobs, **kwargs)

    def fit(self, X, y, EX_list=None, lags=False):
        XEX = _check_assemble(X, EX_list, self.en_nlag, self.ex_nlag, lags)
        return super(KNRLagSelector, self).fit(XEX, y)
    
    def predict(self, X, EX_list=None, lags=False):
        XEX = _check_assemble(X, EX_list, self.en_nlag, self.ex_nlag, lags)
        return super(SVRLagSelector, self).predict(XEX)
    
    def score(self, X, y, EX_list=None, lags=False, sample_weight=None):
        XEX = _check_assemble(X, EX_list, self.en_nlag, self.ex_nlag, lags)
        return super(SVRLagSelector, self).score(XEX, y, sample_weight)




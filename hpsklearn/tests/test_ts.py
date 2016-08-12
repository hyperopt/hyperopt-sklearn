"""
Unit tests for time series forecast using sklearn and hyperopt

In this file, a simulated time series dataset is used to demonstrate the 
use of hpsklearn for time series forecasting problems. More specifically,
it shows: how a time series dataset can be converted into an sklearn 
compatible format; the use of the time series lag selector; and exogenous 
data for training machine learning models.

Briefly, the following formula is used to generate the dataset:

y[t] = a1 * y[t - 1] + a2 * y[t - 2] + 
       b1 * X[t, 1] + b2 * X[t, 2] + b3 * X[t, 3] + 
       c + e
where y is the time series, X is an exogenous dataset, a1, a2, b1, b2, b3 
and c are parameters and e is an error term with the following specifications:

a1 = .666   a2 = -.333  c = -.5
b1 = 1.5    b2 = -1.5   b3 = .5
X[t, 1] ~ uniform(.5, 1.5)
X[t, 2] ~ normal(2., 1.5)
X[t, 3] ~ normal[3., 2.5]
e ~ normal(0, 1.5)

The purpose of learning is to correctly identify the lag size and the values 
of the parameters.

"""
from __future__ import print_function
import sys
try:
    import unittest2 as unittest
except ImportError:
    import unittest
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from hyperopt import tpe
from hpsklearn import HyperoptEstimator, svr_linear
from hpsklearn.components import ts_lagselector


class TimeSeriesForecast(unittest.TestCase):

    def setUp(self):
        '''Generate a simulated dataset and define utility functions
        '''
        ts_size = 1000
        y = np.random.normal(0, 1.5, ts_size)  # white noises, i.e. errors.
        a = np.array([.666, -.333])
        c = -.5
        b = np.array([1.5, -1.5, .5])
        x1 = np.random.uniform(.5, 1.5, ts_size)
        x2 = np.random.normal(2., 1.5, ts_size)
        x3 = np.random.normal(3., 2.5, ts_size)
        X = np.c_[x1, x2, x3]
        for i in range(2, ts_size):
            y[i] += np.dot(a, y[[i - 1, i - 2]])
        y += np.dot(X, b) + c
        self.X = X
        self.y = y
        self.a = a
        self.b = b
        self.c = c
        #### A utility function to convert time series to tabular data ####
        def ts_to_lagged_df(ts, col_id, index=None, lag_size=5):
            '''Convert a time series to lagged dataframe
            Args:
                ts (DataFrame): a dataframe, series or array containing the 
                                time series. If it is a dataframe or array, 
                                the 2nd dimension must have a size of 1.
                col_id (str): if the col id is N, all columns will be named 
                              as N1, N2, N3, ...
                index ([DatetimeIndex]): if not provided, will use the index 
                                         from the input dataframe or series, 
                                         or integers if the input is array.
                lag_size ([int]): the lag size to use
            Returns: a dataframe containing the lagged values in columns.
            '''
            # Sanity check and extract data and index.
            if isinstance(ts, pd.Series):
                dat = ts.values.ravel()
                if index is None:
                    df_idx = ts.index[lag_size:]
            elif isinstance(ts, pd.DataFrame):
                assert ts.shape[1] == 1
                dat = ts.values.ravel()
                if index is None:
                    df_idx = ts.index[lag_size:]
            else:
                assert isinstance(ts, np.ndarray)
                assert ts.ndim <= 2
                if ts.ndim == 2:
                    assert ts.shape[1] == 1
                dat = ts.ravel()
                if index is None:
                    df_idx = range(len(dat) - lag_size)
            if index is not None:
                assert len(index) == len(dat)
                df_idx = index[lag_size:]
            # Convert to lagged dataframe.
            lagged_df = np.concatenate(
                map(lambda x: dat[range(x - 1, x - 1 - lag_size, -1)], 
                    range(lag_size, len(dat)))
            ).reshape((-1, lag_size))
            lagnames = map(lambda x: col_id + str(x), range(1, lag_size + 1))
            lagged_df = pd.DataFrame(lagged_df, index=df_idx, columns=lagnames)
            return lagged_df
        #### End definition ####
        self.ts_to_lagged_df = ts_to_lagged_df


    def test_svm_lagselector(self):
        """
        Using SVM as regressors, I hope hyperopt can help me optimize its 
        hyperparameters and decide the lag size as well. I also have an 
        exogenous dataset that I want to use to help making predictions.

        """
        # Convert time series to tabular.
        max_lag_size = 10
        lagged_y_df = self.ts_to_lagged_df(self.y, 'L', lag_size=max_lag_size)
        # dim of lagged_y_df: [990, 10].
        y_target = self.y[max_lag_size:]
        # Setup train/test predictors and targets.
        test_size = 300
        X_train = lagged_y_df[:-test_size].values
        X_test = lagged_y_df[-test_size:].values
        y_train = y_target[:-test_size]
        y_test = y_target[-test_size:]
        EX_train = self.X[max_lag_size:-test_size, :]
        EX_test = self.X[-test_size:, :]
        # Optimize an SVM for forecasting.
        svr_opt = HyperoptEstimator(
            preprocessing=[ts_lagselector('lag', 1, 10)],
            ex_preprocs=[ [] ],  # must explicitly set EX preprocessings.
            regressor=svr_linear('svm', max_iter=1e5),
            algo=tpe.suggest,
            max_evals=30,
            verbose=True
        )
        svr_opt.fit(X_train, y_train, EX_list=[EX_train],
                    valid_size=.2, cv_shuffle=False)
        # It's generally a good idea to turn off shuffling for time series
        # forecasting to avoid over-optimistic scores.
        ex_n_feas = EX_train.shape[1]
        bm_ = svr_opt.best_model()
        print('\n==== Time series forecast with SVM and lag selectors ====',
              file=sys.stderr)
        print('\nThe best model found:', bm_, file=sys.stderr)
        print('=' * 40, file=sys.stderr)
        print('Actual parameter values\n', 
              'lag size:', len(self.a), '\n',
              'a:', self.a, 'b:', self.b, 'c:', self.c, 
              file=sys.stderr)
        svr_mod = bm_['learner']
        a_hat = np.round(svr_mod.coef_[0, :-ex_n_feas], 3)
        b_hat = np.round(svr_mod.coef_[0, -ex_n_feas:], 3)
        c_hat = np.round(svr_mod.intercept_, 3)
        print('-' * 4, file=sys.stderr)
        print('Estimated parameter values\n',
              'lag size:', len(a_hat), '\n',
              'a:', a_hat, 'b:', b_hat, 'c:', c_hat,
              file=sys.stderr)
        print('=' * 40, file=sys.stderr)
        print('Best trial validation R2:', 
              1 - svr_opt.trials.best_trial['result']['loss'],
              file=sys.stderr)
        print('Test R2:', svr_opt.score(X_test, y_test, EX_list=[EX_test]),
              file=sys.stderr)


if __name__ == '__main__':
    unittest.main()






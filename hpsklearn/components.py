import numpy as np
import sklearn.svm
import sklearn.ensemble
import sklearn.neighbors
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.neural_network
import sklearn.linear_model
import sklearn.feature_extraction.text
import sklearn.naive_bayes
from functools import partial
from hyperopt.pyll import scope, as_apply
from hyperopt import hp
from .vkmeans import ColumnKMeans
from . import lagselectors


@scope.define
def sklearn_SVC(*args, **kwargs):
    return sklearn.svm.SVC(*args, **kwargs)


@scope.define
def sklearn_SVR(*args, **kwargs):
    return sklearn.svm.SVR(*args, **kwargs)

@scope.define
def sklearn_SVRLagSelector(*args, **kwargs):
    return lagselectors.SVRLagSelector(*args, **kwargs)

@scope.define
def sklearn_KNRLagSelector(*args, **kwargs):
    return lagselectors.KNRLagSelector(*args, **kwargs)

@scope.define
def sklearn_RFRLagSelector(*args, **kwargs):
    return lagselectors.RFRLagSelector(*args, **kwargs)

@scope.define
def sklearn_ETRLagSelector(*args, **kwargs):
    return lagselectors.ETRLagSelector(*args, **kwargs)

@scope.define
def sklearn_LinearSVC(*args, **kwargs):
    return sklearn.svm.LinearSVC(*args, **kwargs)


@scope.define
def sklearn_KNeighborsClassifier(*args, **kwargs):
    return sklearn.neighbors.KNeighborsClassifier(*args, **kwargs)


@scope.define
def sklearn_KNeighborsRegressor(*args, **kwargs):
    return sklearn.neighbors.KNeighborsRegressor(*args, **kwargs)


@scope.define
def sklearn_RandomForestClassifier(*args, **kwargs):
    return sklearn.ensemble.RandomForestClassifier(*args, **kwargs)


@scope.define
def sklearn_RandomForestRegressor(*args, **kwargs):
    return sklearn.ensemble.RandomForestRegressor(*args, **kwargs)


@scope.define
def sklearn_ExtraTreesClassifier(*args, **kwargs):
    return sklearn.ensemble.ExtraTreesClassifier(*args, **kwargs)


@scope.define
def sklearn_ExtraTreesRegressor(*args, **kwargs):
    return sklearn.ensemble.ExtraTreesRegressor(*args, **kwargs)


@scope.define
def sklearn_SGDClassifier(*args, **kwargs):
    return sklearn.linear_model.SGDClassifier(*args, **kwargs)


@scope.define
def sklearn_SGDRegressor(*args, **kwargs):
    return sklearn.linear_model.SGDRegressor(*args, **kwargs)

# @scope.define
# def sklearn_Ridge(*args, **kwargs):
#     return sklearn.linear_model.Ridge(*args, **kwargs)


@scope.define
def sklearn_MultinomialNB(*args, **kwargs):
    return sklearn.naive_bayes.MultinomialNB(*args, **kwargs)


@scope.define
def sklearn_PCA(*args, **kwargs):
    return sklearn.decomposition.PCA(*args, **kwargs)


@scope.define
def sklearn_Tfidf(*args, **kwargs):
    return sklearn.feature_extraction.text.TfidfVectorizer(*args, **kwargs)


@scope.define
def sklearn_StandardScaler(*args, **kwargs):
    return sklearn.preprocessing.StandardScaler(*args, **kwargs)


@scope.define
def sklearn_MinMaxScaler(*args, **kwargs):
    return sklearn.preprocessing.MinMaxScaler(*args, **kwargs)


@scope.define
def sklearn_Normalizer(*args, **kwargs):
    return sklearn.preprocessing.Normalizer(*args, **kwargs)


@scope.define
def sklearn_OneHotEncoder(*args, **kwargs):
    return sklearn.preprocessing.OneHotEncoder(*args, **kwargs)


@scope.define
def sklearn_BernoulliRBM(*args, **kwargs):
    return sklearn.neural_network.BernoulliRBM(*args, **kwargs)


@scope.define
def sklearn_ColumnKMeans(*args, **kwargs):
    return ColumnKMeans(*args, **kwargs)


@scope.define
def patience_param(x):
    """
    Mark a hyperparameter as having a simple monotonic increasing
    relationship with both CPU time and the goodness of the model.
    """
    # -- TODO: make this do something!
    return x


@scope.define
def inv_patience_param(x):
    """
    Mark a hyperparameter as having a simple monotonic decreasing
    relationship with both CPU time and the goodness of the model.
    """
    # -- TODO: make this do something!
    return x


def hp_bool(name):
    return hp.choice(name, [False, True])


_svm_default_cache_size = 512


def _svm_gamma(name, n_features=1):
    '''Generator of default gamma values for SVMs.
    This setting is based on the following rationales:
    1.  The gamma hyperparameter is basically an amplifier for the 
        original dot product or l2 norm.
    2.  The original dot product or l2 norm shall be normalized by 
        the number of features first.
    '''
    # -- making these non-conditional variables
    #    probably helps the GP algorithm generalize
    # assert n_features >= 1
    return hp.loguniform(name,
                         np.log(1. / n_features * 1e-3),
                         np.log(1. / n_features * 1e3))


def _svm_degree(name):
    return hp.quniform(name, 1.5, 6.5, 1)


def _svm_max_iter(name):
    return hp.qloguniform(name, np.log(1e7), np.log(1e9), 1)


def _svm_C(name):
    return hp.loguniform(name, np.log(1e-5), np.log(1e5))


def _svm_tol(name):
    return hp.loguniform(name, np.log(1e-5), np.log(1e-2))


def _svm_int_scaling(name):
    return hp.loguniform(name, np.log(1e-1), np.log(1e1))


def _svm_epsilon(name):
    return hp.loguniform(name, np.log(1e-3), np.log(1e3))


def _svm_loss_penalty_dual(name):
    """
    The combination of penalty='l1' and loss='hinge' is not supported
    penalty='l2' and loss='hinge' is only supported when dual='true'
    penalty='l1' is only supported when dual='false'.
    """
    return hp.choice(name, [
        ('hinge', 'l2', True),
        ('squared_hinge', 'l2', True),
        ('squared_hinge', 'l1', False),
        ('squared_hinge', 'l2', False)
    ])


def _knn_metric_p(name, sparse_data=False, metric=None, p=None):
    if sparse_data:
        return ('euclidean', 2)
    elif metric == 'euclidean':
        return (metric, 2)
    elif metric == 'manhattan':
        return (metric, 1)
    elif metric == 'chebyshev':
        return (metric, 0)
    elif metric == 'minkowski':
        assert p is not None
        return (metric, p)
    elif metric is None:
        return hp.pchoice(name, [
            (0.55, ('euclidean', 2)),
            (0.15, ('manhattan', 1)),
            (0.15, ('chebyshev', 0)),
            (0.15, ('minkowski', _knn_p(name + '.p'))),
        ])
    else:
        return (metric, p)  # undefined, simply return user input.


def _knn_p(name):
    return hp.quniform(name, 2.5, 5.5, 1)


def _knn_neighbors(name):
    return scope.int(hp.qloguniform(name, np.log(0.5), np.log(50.5), 1))


def _knn_weights(name):
    return hp.choice(name, ['uniform', 'distance'])


def _trees_n_estimators(name):
    return scope.int(hp.qloguniform(name, np.log(9.5), np.log(3000.5), 1))

def _trees_criterion(name):
    return hp.choice(name, ['gini', 'entropy'])


def _trees_max_features(name):
    return hp.pchoice(name, [
        (0.2, 'sqrt'),  # most common choice.
        (0.1, 'log2'),  # less common choice.
        (0.1, None),  # all features, less common choice.
        (0.6, hp.uniform(name + '.frac', 0., 1.))
    ])

def _trees_max_depth(name):
    return hp.pchoice(name, [
        (0.7, None),  # most common choice.
        # Try some shallow trees.
        (0.1, 2),
        (0.1, 3),
        (0.1, 4),
    ])


def _trees_min_samples_split(name):
    return 2


def _trees_min_samples_leaf(name):
    return hp.choice(name, [
        1,  # most common choice.
        scope.int(hp.qloguniform(name + '.gt1', np.log(1.5), np.log(50.5), 1))
    ])


def _trees_bootstrap(name):
    return hp.choice(name, [True, False])


def _sgd_penalty(name):
    return hp.pchoice(name, [
        (0.40, 'l2'),
        (0.35, 'l1'),
        (0.25, 'elasticnet')
    ])


def _sgd_alpha(name):
    return hp.loguniform(name, np.log(1e-6), np.log(1e-1))


def _sgd_l1_ratio(name):
    return hp.uniform(name, 0, 1)


def _sgd_epsilon(name):
    return hp.loguniform(name, np.log(1e-7), np.log(1))


def _sgdc_learning_rate(name):
    return hp.pchoice(name, [
        (0.50, 'optimal'),
        (0.25, 'invscaling'),
        (0.25, 'constant')
    ])


def _sgdr_learning_rate(name):
    return hp.pchoice(name, [
        (0.50, 'invscaling'),
        (0.25, 'optimal'),
        (0.25, 'constant')
    ])


def _sgd_eta0(name):
    return hp.loguniform(name, np.log(1e-5), np.log(1e-1))


def _sgd_power_t(name):
    return hp.uniform(name, 0, 1)


def _random_state(name, random_state):
    if random_state is None:
        return hp.randint(name, 5)
    else:
        return random_state

def _class_weight(name):
    return hp.choice(name, [None, 'balanced'])

def _lag_size_en(name, max_lag_size):
    return scope.int(hp.quniform(name, 0.5, max_lag_size + 0.5, 1))

def _lag_size_ex(name, max_lag_size):
    # Lag size can be zero for exogenous data.
    return scope.int(hp.quniform(name, -0.5, max_lag_size + 0.5, 1))

def _determine_lag_sizes(name, max_lag_sizes, n_ex_ds, en_nlag, ex_nlag):
    if max_lag_sizes is None:
        assert (en_nlag is not None and ex_nlag is not None)
        return (en_nlag, ex_nlag)
    else:
        assert (n_ex_ds is not None and n_ex_ds >= 0)
        if isinstance(max_lag_sizes, (list, tuple)):
            assert len(max_lag_sizes) == n_ex_ds + 1
        else:
            assert isinstance(max_lag_sizes, int)
        if en_nlag is not None:
            en_nlag = en_nlag
        elif isinstance(max_lag_sizes, int):
            en_nlag = _lag_size_en(name + '.en', max_lag_sizes) 
        else:
            en_nlag = _lag_size_en(name + '.en', max_lag_sizes[0])
        if ex_nlag is not None:
            ex_nlag = ex_nlag
        elif n_ex_ds == 0:
            ex_nlag = [0]
        elif isinstance(max_lag_sizes, int):
            ex_nlag = [
                _lag_size_ex(name + '.ex' + str(i), 
                             max_lag_sizes) for i in range(1, n_ex_ds + 1)
            ]
        else:
            ex_nlag = [
                _lag_size_ex(name + '.ex' + str(i), 
                             max_lag_sizes[i]) for i in range(1, n_ex_ds + 1)
            ]
        return (en_nlag, ex_nlag)


def _svm_hp_space(
        name_func,
        kernel,
        n_features=1,
        C=None,
        gamma=None,
        coef0=None,
        degree=None,
        shrinking=None,
        tol=None,
        max_iter=None,
        verbose=False,
        cache_size=_svm_default_cache_size):
    '''Generate SVM hyperparamters search space
    '''
    if kernel in ['linear', 'rbf', 'sigmoid']:
        degree_ = 1
    else:
        degree_ = (_svm_degree(name_func('degree')) 
                   if degree is None else degree)
    if kernel in ['linear']:
        gamma_ = 'auto'
    else:
        gamma_ = (_svm_gamma(name_func('gamma'), n_features=1) 
                  if gamma is None else gamma)
        gamma_ /= n_features  # make gamma independent on lag sizes.
    if kernel in ['linear', 'rbf']:
        coef0_ = 0.0
    elif coef0 is None:
        if kernel == 'poly':
            coef0_ = hp.pchoice(name_func('coef0'), [
                (0.3, 0),
                (0.7, gamma_ * hp.uniform(name_func('coef0val'), 0., 10.))
            ])
        elif kernel == 'sigmoid':
            coef0_ = hp.pchoice(name_func('coef0'), [
                (0.3, 0),
                (0.7, gamma_ * hp.uniform(name_func('coef0val'), -10., 10.))
            ])
        else:
            pass
    else:
        coef0_ = coef0

    hp_space = dict(
        kernel=kernel,
        C=_svm_C(name_func('C')) if C is None else C,
        gamma=gamma_,
        coef0=coef0_,
        degree=degree_,
        shrinking=(hp_bool(name_func('shrinking')) 
                   if shrinking is None else shrinking),
        tol=_svm_tol(name_func('tol')) if tol is None else tol,
        max_iter=(_svm_max_iter(name_func('maxiter'))
                  if max_iter is None else max_iter),
        verbose=verbose,
        cache_size=cache_size)
    return hp_space

def _svc_hp_space(name_func, random_state=None):
    '''Generate SVC specific hyperparamters
    '''
    hp_space['random_state'] = _random_state(name_func('rstate'), 
                                             random_state)
    return hp_space

def _svr_hp_space(name_func, epsilon=None):
    '''Generate SVR specific hyperparamters
    '''
    hp_space = {}
    hp_space['epsilon'] = (_svm_epsilon(name_func('epsilon')) 
                           if epsilon is None else epsilon)
    return hp_space

def _lags_hp_space(
        name_func,
        max_lag_sizes=None,
        n_ex_ds=None,
        en_nlag=None,
        ex_nlag=None):
    '''Generate lag selector hyperparamters
    Args:
        name_func (callable): a function to generate names for hyperparamters
        max_lag_sizes([int or list or tuple]): maximum lag size(s)
        n_ex_ds ([int]): number of exogenous datasets
        en_nlag ([int]): number of lags for endogenous dataset
        ex_nlag ([list or tuple]): number(s) of lags for exogenous datasets

    max_lag_sizes, n_ex_ds, en_nlag, ex_nlag jointly determine the 
    lag sizes for endogenous and exogenous predictors.

    If max_lag_sizes is none, en_nlag and ex_nlag shall both be specified.
    Otherwise, max_lag_sizes gives the maximum lag size(s) for the 
    predictors to choose from. If max_lag_sizes is int, the same value is 
    used for all predictors. If max_lag_sizes is a list, different values 
    can be used for the predictors separately. max_lag_sizes[0] shall contain
    the value for endogenous predictors, max_lag_sizes[1:] shall contain 
    the values for exogenous predictors and use the same order as EX_list.

    For endogenous lag size, the default distribution is to choose from 
    uniform(1, max_lag_size). While for exogenous lag size, the default 
    is to choose from uniform(0, max_lag_size). The lag size(s) must be 
    integer(s). Custom values can be provided for en_nlag and ex_nlag to 
    override this behavior.

    When max_lag_sizes and en_nlag are both specified, en_nlag will override 
    the value in max_lag_sizes. The same for ex_nlag.

    n_ex_ds gives the number of exogenous datasets. It must be specified 
    when max_lag_sizes is specified. If there is no exogenous data, it can 
    simply be set to 0.     
    '''
    en_nlag_, ex_nlag_ = _determine_lag_sizes(name_func('lag_size'), 
                                              max_lag_sizes, n_ex_ds, 
                                              en_nlag, ex_nlag)
    hp_space = {}
    hp_space['en_nlag'] = en_nlag_
    hp_space['ex_nlag'] = ex_nlag_
    return hp_space

def svr_lags_kernel(
        name,
        kernel,
        max_lag_sizes=None,
        n_ex_ds=None,
        en_nlag=None,
        ex_nlag=None,
        C=None,
        epsilon=None,
        gamma=None,
        coef0=None,
        degree=None,
        shrinking=None,
        tol=None,
        max_iter=None,
        verbose=False,
        cache_size=_svm_default_cache_size):
    '''
    Return a pyll graph with hyperparamters that will construct
    a lagselectors.SVRLagSelector model with user specified kernel.

    See help(hpsklearn.components._lags_hp_space) for details on 
    specifying lag sizes.
    '''
    def _name(msg):
        return '%s.%s_%s' % (name, kernel, msg)
    # Lag selector hyperparameters.
    hp_space = _lags_hp_space(_name, 
                              max_lag_sizes=max_lag_sizes, 
                              n_ex_ds=n_ex_ds,
                              en_nlag=en_nlag,
                              ex_nlag=ex_nlag)
    # SVM hyperparameters.
    n_features = scope.int(sum([hp_space['en_nlag']] + hp_space['ex_nlag']))
    hp_space.update(_svm_hp_space(_name, kernel=kernel,
                                  n_features=n_features,
                                  C=C,
                                  gamma=gamma,
                                  coef0=coef0,
                                  degree=degree,
                                  shrinking=shrinking,
                                  tol=tol,
                                  max_iter=max_iter,
                                  verbose=verbose,
                                  cache_size=cache_size))
    # SVR specific hyperparameters.
    hp_space.update(_svr_hp_space(_name, epsilon=epsilon))

    return scope.sklearn_SVRLagSelector(**hp_space)


def svr_lags_linear(name, **kwargs):
    '''
    This simply return the result of svr_lags_kernel with the 
    kernel fixed as linear.
    See help(svr_lags_kernel) for details.
    '''
    return svr_lags_kernel(name, kernel='linear', **kwargs)

def svr_lags_rbf(name, **kwargs):
    '''
    This simply return the result of svr_lags_kernel with the 
    kernel fixed as rbf.
    See help(svr_lags_kernel) for details.
    '''
    return svr_lags_kernel(name, kernel='rbf', **kwargs)

def svr_lags_poly(name, **kwargs):
    '''
    This simply return the result of svr_lags_kernel with the 
    kernel fixed as poly.
    See help(svr_lags_kernel) for details.
    '''
    return svr_lags_kernel(name, kernel='poly', **kwargs)

def svr_lags_sigmoid(name, **kwargs):
    '''
    This simply return the result of svr_lags_kernel with the 
    kernel fixed as sigmoid.
    See help(svr_lags_kernel) for details.
    '''
    return svr_lags_kernel(name, kernel='sigmoid', **kwargs)


def svr_lags(
        name, 
        kernels=['linear', 'rbf', 'poly', 'sigmoid'], 
        **kwargs):
    '''SVR lag selector with kernel function to be chosen by hyperopt
    Args:
        kernels ([list]): available kernels to choose from.
        **kwargs: all other parameters for svr_lagselector.
    Return: an SVRLagSelector instance.
    '''
    svms = {
        'linear': partial(svr_lags_linear, name=name),
        'rbf': partial(svr_lags_rbf, name=name),
        'poly': partial(svr_lags_poly, name=name),
        'sigmoid': partial(svr_lags_sigmoid, name=name),
    }
    choices = [svms[kern](**kwargs) for kern in kernels]
    if len(choices) == 1:
        rval = choices[0]
    else:
        rval = hp.choice('%s.kernel' % name, choices)
    return rval


def svc_linear(name,
               C=None,
               shrinking=None,
               tol=None,
               max_iter=None,
               verbose=False,
               random_state=None,
               cache_size=_svm_default_cache_size):
    """
    Return a pyll graph with hyperparamters that will construct
    a sklearn.svm.SVC model with a linear kernel.

    """
    def _name(msg):
        return '%s.%s_%s' % (name, 'linear', msg)

    rval = scope.sklearn_SVC(
        kernel='linear',
        C=_svm_C(_name('C')) if C is None else C,
        shrinking=hp_bool(
            _name('shrinking')) if shrinking is None else shrinking,
        tol=_svm_tol(_name('tol')) if tol is None else tol,
        max_iter=(_svm_max_iter(_name('maxiter'))
                  if max_iter is None else max_iter),
        verbose=verbose,
        random_state=_random_state(_name('rstate'), random_state),
        cache_size=cache_size,
    )
    return rval


def svr_linear(name,
               C=None,
               epsilon=None,
               shrinking=None,
               tol=None,
               max_iter=None,
               verbose=False,
               cache_size=_svm_default_cache_size):
    """
    Return a pyll graph with hyperparamters that will construct
    a sklearn.svm.SVR model with a linear kernel.

    """
    def _name(msg):
        return '%s.%s_%s' % (name, 'linear', msg)

    rval = scope.sklearn_SVR(
        kernel='linear',
        C=_svm_C(_name('C')) if C is None else C,
        epsilon=_svm_epsilon(_name('epsilon')) if epsilon is None else epsilon,
        shrinking=hp_bool(
            _name('shrinking')) if shrinking is None else shrinking,
        tol=_svm_tol(_name('tol')) if tol is None else tol,
        max_iter=(_svm_max_iter(_name('maxiter'))
                  if max_iter is None else max_iter),
        verbose=verbose,
        # random_state=_random_state(_name('rstate'), random_state),
        cache_size=cache_size,
    )
    return rval


def svc_rbf(name,
            n_features=1,
            C=None,
            gamma=None,
            shrinking=None,
            tol=None,
            max_iter=None,
            verbose=False,
            random_state=None,
            cache_size=_svm_default_cache_size):
    """
    Return a pyll graph with hyperparamters that will construct
    a sklearn.svm.SVC model with an RBF kernel.

    """
    def _name(msg):
        return '%s.%s_%s' % (name, 'rbf', msg)

    rval = scope.sklearn_SVC(
        kernel='rbf',
        C=_svm_C(_name('C')) if C is None else C,
        gamma=(_svm_gamma(_name('gamma'), n_features)
               if gamma is None else gamma),
        shrinking=hp_bool(
            _name('shrinking')) if shrinking is None else shrinking,
        tol=_svm_tol(_name('tol')) if tol is None else tol,
        max_iter=(_svm_max_iter(_name('maxiter'))
                  if max_iter is None else max_iter),
        verbose=verbose,
        cache_size=cache_size,
        random_state=_random_state(_name('rstate'), random_state),
    )
    return rval


def svr_rbf(name,
            n_features=1,
            C=None,
            epsilon=None,
            gamma=None,
            shrinking=None,
            tol=None,
            max_iter=None,
            verbose=False,
            cache_size=_svm_default_cache_size):
    """
    Return a pyll graph with hyperparamters that will construct
    a sklearn.svm.SVR model with an RBF kernel.

    """
    def _name(msg):
        return '%s.%s_%s' % (name, 'rbf', msg)

    rval = scope.sklearn_SVR(
        kernel='rbf',
        C=_svm_C(_name('C')) if C is None else C,
        epsilon=_svm_epsilon(_name('epsilon')) if epsilon is None else epsilon,
        gamma=(_svm_gamma(_name('gamma'), n_features)
               if gamma is None else gamma),
        shrinking=hp_bool(
            _name('shrinking')) if shrinking is None else shrinking,
        tol=_svm_tol(_name('tol')) if tol is None else tol,
        max_iter=(_svm_max_iter(_name('maxiter'))
                  if max_iter is None else max_iter),
        verbose=verbose,
        cache_size=cache_size,
        # random_state=_random_state(_name('rstate'), random_state),
    )
    return rval


def svc_poly(name,
             n_features=1,
             C=None,
             gamma=None,
             coef0=None,
             degree=None,
             shrinking=None,
             tol=None,
             max_iter=None,
             verbose=False,
             random_state=None,
             cache_size=_svm_default_cache_size):
    """
    Return a pyll graph with hyperparamters that will construct
    a sklearn.svm.SVC model with an RBF kernel.

    """
    def _name(msg):
        return '%s.%s_%s' % (name, 'poly', msg)

    # Some changes to the original design:
    # 1. The gamma hyperparameter scales the dot product, so shall scale
    #    the coef0 as well.
    # 2. It is more likely to shift the dot product to the positive side
    #    so that all of them will become > 1.0.
    # -- (K(x, y) + coef0)^d
    poly_gamma = (_svm_gamma(_name('gamma'), n_features)
                  if gamma is None else gamma)
    poly_coef0 = hp.pchoice(_name('coef0'), [
        (0.3, 0),
        (0.7, poly_gamma * hp.uniform(_name('coef0val'), 0., 10.))
    ]) if coef0 is None else coef0

    rval = scope.sklearn_SVC(
        kernel='poly',
        C=_svm_C(_name('C')) if C is None else C,
        gamma=poly_gamma,
        coef0=poly_coef0,
        degree=_svm_degree(_name('degree')) if degree is None else degree,
        shrinking=(hp_bool(_name('shrinking'))
                   if shrinking is None else shrinking),
        tol=_svm_tol(_name('tol')) if tol is None else tol,
        max_iter=(_svm_max_iter(_name('maxiter'))
                  if max_iter is None else max_iter),
        verbose=verbose,
        random_state=_random_state(_name('rstate'), random_state),
        cache_size=cache_size,
    )
    return rval


def svr_poly(name,
             n_features=1,
             C=None,
             epsilon=None,
             gamma=None,
             coef0=None,
             degree=None,
             shrinking=None,
             tol=None,
             max_iter=None,
             verbose=False,
             cache_size=_svm_default_cache_size):
    """
    Return a pyll graph with hyperparamters that will construct
    a sklearn.svm.SVR model with an RBF kernel.

    """
    def _name(msg):
        return '%s.%s_%s' % (name, 'poly', msg)

    # Some changes to the original design:
    # 1. The gamma hyperparameter scales the dot product, so shall scale
    #    the coef0 as well.
    # 2. It is more likely to shift the dot product to the positive side
    #    so that all of them will become > 1.0.
    # -- (K(x, y) + coef0)^d
    poly_gamma = (_svm_gamma(_name('gamma'), n_features)
                  if gamma is None else gamma)
    poly_coef0 = hp.pchoice(_name('coef0'), [
        (0.3, 0),
        (0.7, poly_gamma * hp.uniform(_name('coef0val'), 0., 10.))
    ]) if coef0 is None else coef0

    rval = scope.sklearn_SVR(
        kernel='poly',
        C=_svm_C(_name('C')) if C is None else C,
        epsilon=_svm_epsilon(_name('epsilon')) if epsilon is None else epsilon,
        gamma=poly_gamma,
        coef0=poly_coef0,
        degree=_svm_degree(_name('degree')) if degree is None else degree,
        shrinking=(hp_bool(_name('shrinking'))
                   if shrinking is None else shrinking),
        tol=_svm_tol(_name('tol')) if tol is None else tol,
        max_iter=(_svm_max_iter(_name('maxiter'))
                  if max_iter is None else max_iter),
        verbose=verbose,
        # random_state=_random_state(_name('rstate'), random_state),
        cache_size=cache_size,
    )
    return rval


def svc_sigmoid(name,
                n_features=1,
                C=None,
                gamma=None,
                coef0=None,
                shrinking=None,
                tol=None,
                max_iter=None,
                verbose=False,
                random_state=None,
                cache_size=_svm_default_cache_size):
    """
    Return a pyll graph with hyperparamters that will construct
    a sklearn.svm.SVC model with an RBF kernel.

    """
    def _name(msg):
        return '%s.%s_%s' % (name, 'sigmoid', msg)

    # Some changes to the original design:
    # 1. The gamma hyperparameter scales the dot product, so shall scale
    #    the coef0 as well.
    # 2. The purpose of the tanh function is for saturation. There is no
    #    preference for positive or negative activation. So the coef0 is
    #    set to sample from a symmetric range.
    # -- tanh(K(x, y) + coef0)
    sigm_gamma = (_svm_gamma(_name('gamma'), n_features)
                  if gamma is None else gamma)
    sigm_coef0 = hp.pchoice(_name('coef0'), [
        (0.3, 0),
        (0.7, sigm_gamma * hp.uniform(_name('coef0val'), -10., 10.))
    ]) if coef0 is None else coef0

    rval = scope.sklearn_SVC(
        kernel='sigmoid',
        C=_svm_C(_name('C')) if C is None else C,
        gamma=sigm_gamma,
        coef0=sigm_coef0,
        shrinking=hp_bool(
            _name('shrinking')) if shrinking is None else shrinking,
        tol=_svm_tol(_name('tol')) if tol is None else tol,
        max_iter=(_svm_max_iter(_name('maxiter'))
                  if max_iter is None else max_iter),
        verbose=verbose,
        random_state=_random_state(_name('rstate'), random_state),
        cache_size=cache_size)
    return rval


def svr_sigmoid(name,
                n_features=1,
                C=None,
                epsilon=None,
                gamma=None,
                coef0=None,
                shrinking=None,
                tol=None,
                max_iter=None,
                verbose=False,
                cache_size=_svm_default_cache_size):
    """
    Return a pyll graph with hyperparamters that will construct
    a sklearn.svm.SVR model with an RBF kernel.

    """
    def _name(msg):
        return '%s.%s_%s' % (name, 'sigmoid', msg)

    # Some changes to the original design:
    # 1. The gamma hyperparameter scales the dot product, so shall scale
    #    the coef0 as well.
    # 2. The purpose of the tanh function is for saturation. There is no
    #    preference for positive or negative activation. So the coef0 is
    #    set to sample from a symmetric range.
    # -- tanh(K(x, y) + coef0)
    sigm_gamma = (_svm_gamma(_name('gamma'), n_features)
                  if gamma is None else gamma)
    sigm_coef0 = hp.pchoice(_name('coef0'), [
        (0.3, 0),
        (0.7, sigm_gamma * hp.uniform(_name('coef0val'), -10., 10.))
    ]) if coef0 is None else coef0

    rval = scope.sklearn_SVR(
        kernel='sigmoid',
        C=_svm_C(_name('C')) if C is None else C,
        epsilon=_svm_epsilon(_name('epsilon')) if epsilon is None else epsilon,
        gamma=sigm_gamma,
        coef0=sigm_coef0,
        shrinking=hp_bool(
            _name('shrinking')) if shrinking is None else shrinking,
        tol=_svm_tol(_name('tol')) if tol is None else tol,
        max_iter=(_svm_max_iter(_name('maxiter'))
                  if max_iter is None else max_iter),
        verbose=verbose,
        # random_state=_random_state(_name('rstate'), random_state),
        cache_size=cache_size)
    return rval


def svc(name,
        n_features=1,
        C=None,
        kernels=['linear', 'rbf', 'poly', 'sigmoid'],
        shrinking=None,
        tol=None,
        max_iter=None,
        verbose=False,
        random_state=None,
        cache_size=_svm_default_cache_size):
    svms = {
        'linear': svc_linear(
            name,
            C=C,
            shrinking=shrinking,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            verbose=verbose),
        'rbf': svc_rbf(
            name,
            n_features=n_features,
            C=C,
            shrinking=shrinking,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            verbose=verbose),
        'poly': svc_poly(
            name,
            n_features=n_features,
            C=C,
            shrinking=shrinking,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            verbose=verbose),
        'sigmoid': svc_sigmoid(
            name,
            n_features=n_features,
            C=C,
            shrinking=shrinking,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            verbose=verbose),
    }
    choices = [svms[kern] for kern in kernels]
    if len(choices) == 1:
        rval = choices[0]
    else:
        rval = hp.choice('%s.kernel' % name, choices)
    return rval


def svr(name,
        n_features=1,
        C=None,
        epsilon=None,
        kernels=['linear', 'rbf', 'poly', 'sigmoid'],
        shrinking=None,
        tol=None,
        max_iter=None,
        verbose=False,
        cache_size=_svm_default_cache_size):
    svms = {
        'linear': svr_linear(
            name,
            C=C,
            epsilon=epsilon,
            shrinking=shrinking,
            tol=tol,
            max_iter=max_iter,
            # random_state=random_state,
            verbose=verbose),
        'rbf': svr_rbf(
            name,
            n_features=n_features,
            C=C,
            epsilon=epsilon,
            shrinking=shrinking,
            tol=tol,
            max_iter=max_iter,
            # random_state=random_state,
            verbose=verbose),
        'poly': svr_poly(
            name,
            n_features=n_features,
            C=C,
            epsilon=epsilon,
            shrinking=shrinking,
            tol=tol,
            max_iter=max_iter,
            # random_state=random_state,
            verbose=verbose),
        'sigmoid': svr_sigmoid(
            name,
            n_features=n_features,
            C=C,
            epsilon=epsilon,
            shrinking=shrinking,
            tol=tol,
            max_iter=max_iter,
            # random_state=random_state,
            verbose=verbose),
    }
    choices = [svms[kern] for kern in kernels]
    if len(choices) == 1:
        rval = choices[0]
    else:
        rval = hp.choice('%s.kernel' % name, choices)
    return rval


# TODO: Some combinations of parameters are not allowed in LinearSVC
def liblinear_svc(name,
                  C=None,
                  loss=None,
                  penalty=None,
                  dual=None,
                  tol=None,
                  multi_class=None,
                  fit_intercept=True,
                  intercept_scaling=None,
                  class_weight='choose',
                  random_state=None,
                  verbose=False,
                  max_iter=1000):

    def _name(msg):
        return '%s.%s_%s' % (name, 'linear_svc', msg)

    loss_penalty_dual = _svm_loss_penalty_dual(_name('loss_penalty_dual'))

    rval = scope.sklearn_LinearSVC(
        C=_svm_C(_name('C')) if C is None else C,
        loss=loss_penalty_dual[0] if loss is None else loss,
        penalty=loss_penalty_dual[1] if penalty is None else penalty,
        dual=loss_penalty_dual[2] if dual is None else dual,
        tol=_svm_tol(_name('tol')) if tol is None else tol,
        multi_class=(hp.choice(_name('multiclass'), ['ovr', 'crammer_singer'])
                     if multi_class is None else multi_class),
        fit_intercept=fit_intercept,
        intercept_scaling=(_svm_int_scaling(_name('intscaling'))
                           if intercept_scaling is None else intercept_scaling),
        class_weight=(_class_weight(_name('clsweight'))
                      if class_weight == 'choose' else class_weight),
        random_state=_random_state(_name('rstate'), random_state),
        verbose=verbose,
        max_iter=max_iter,
    )
    return rval

def _knn_hp_space(
        name_func,
        sparse_data=False,
        n_neighbors=None,
        weights=None,
        algorithm='auto',
        leaf_size=30,
        metric=None,
        p=None,
        metric_params=None,
        n_jobs=1):
    '''Generate KNN hyperparameters search space
    '''
    metric_p = _knn_metric_p(name_func('metric_p'), sparse_data, metric, p)
    hp_space = dict(
        n_neighbors=(_knn_neighbors(name_func('neighbors'))
                     if n_neighbors is None else n_neighbors),
        weights=(_knn_weights(name_func('weights')) 
                 if weights is None else weights),
        algorithm=algorithm,
        leaf_size=leaf_size,
        metric=metric_p[0],
        p=metric_p[1],
        metric_params=metric_params,
        n_jobs=n_jobs)
    return hp_space

def knr_lags(
        name,
        max_lag_sizes=None,
        n_ex_ds=None,
        en_nlag=None,
        ex_nlag=None,
        sparse_data=False,
        n_neighbors=None,
        weights=None,
        algorithm='auto',
        leaf_size=30,
        metric=None,
        p=None,
        metric_params=None,
        n_jobs=1):
    '''
    Return a pyll graph with hyperparamters that will construct
    a lagselectors.KNRLagSelector (KNN regression lag selector) 
    model.

    See help(hpsklearn.components._lags_hp_space) for details on 
    specifying lag sizes.
    '''
    def _name(msg):
        return '%s.%s_%s' % (name, 'knr', msg)
    # Lag selector hyperparameters.
    hp_space = _lags_hp_space(_name, 
                              max_lag_sizes=max_lag_sizes, 
                              n_ex_ds=n_ex_ds,
                              en_nlag=en_nlag,
                              ex_nlag=ex_nlag)
    # KNN hyperparameters.
    hp_space.update(_knn_hp_space(_name,
                                  sparse_data=sparse_data,
                                  n_neighbors=n_neighbors,
                                  weights=weights,
                                  algorithm=algorithm,
                                  leaf_size=leaf_size,
                                  metric=metric,
                                  p=p,
                                  metric_params=metric_params,
                                  n_jobs=n_jobs))
    return scope.sklearn_KNRLagSelector(**hp_space)


# TODO: Pick reasonable default values
def knn(name,
        sparse_data=False,
        n_neighbors=None,
        weights=None,
        algorithm='auto',
        leaf_size=30,
        metric=None,
        p=None,
        metric_params=None,
        n_jobs=1):

    def _name(msg):
        return '%s.%s_%s' % (name, 'knc', msg)

    rval = scope.sklearn_KNeighborsClassifier(
        n_neighbors=(_knn_neighbors(_name('neighbors'))
                     if n_neighbors is None else n_neighbors),
        weights=_knn_weights(_name('weights')) if weights is None else weights,
        algorithm=algorithm,
        leaf_size=leaf_size,
        metric=(_knn_metric(_name('metric'), sparse_data)
                if metric is None else metric),
        p=_knn_p(_name('p')) if p is None else p,
        metric_params=metric_params,
        n_jobs=n_jobs,
    )
    return rval

# TODO: Pick reasonable default values


def knn_regression(name,
                   sparse_data=False,
                   n_neighbors=None,
                   weights=None,
                   algorithm='auto',
                   leaf_size=30,
                   metric=None,
                   p=None,
                   metric_params=None,
                   n_jobs=1):

    def _name(msg):
        return '%s.%s_%s' % (name, 'knr', msg)

    rval = scope.sklearn_KNeighborsRegressor(
        n_neighbors=(_knn_neighbors(_name('neighbors'))
                     if n_neighbors is None else n_neighbors),
        weights=_knn_weights(_name('weights')) if weights is None else weights,
        algorithm=algorithm,
        leaf_size=leaf_size,
        metric=(_knn_metric(_name('metric'), sparse_data)
                if metric is None else metric),
        p=_knn_p(_name('p')) if p is None else p,
        metric_params=metric_params,
        n_jobs=n_jobs,
    )
    return rval

def _trees_hp_space(
        name_func,
        n_estimators=None,
        criterion='mse',
        max_features=None,
        max_depth=None,
        min_samples_split=None,
        min_samples_leaf=None,
        bootstrap=None,
        oob_score=False,
        n_jobs=1,
        random_state=None,
        verbose=False):
    '''Generate trees ensemble hyperparameters search space
    '''
    hp_space = dict(
        n_estimators=(_trees_n_estimators(name_func('n_estimators')) 
                      if n_estimators is None else n_estimators),
        criterion=criterion,
        max_features=(_trees_max_features(name_func('max_features'))
                      if max_features is None else max_features),
        max_depth=(_trees_max_depth(name_func('max_depth'))
                   if max_depth is None else max_depth),
        min_samples_split=(_trees_min_samples_split(name_func('min_samples_split'))
                           if min_samples_split is None else min_samples_split),
        min_samples_leaf=(_trees_min_samples_leaf(name_func('min_samples_leaf'))
                          if min_samples_leaf is None else min_samples_leaf),
        bootstrap=(_trees_bootstrap(name_func('bootstrap'))
                   if bootstrap is None else bootstrap),
        oob_score=oob_score,
        n_jobs=n_jobs,
        random_state=_random_state(name_func('rstate'), random_state),
        verbose=verbose,
    )
    return hp_space

def rfr_lags(
        name,
        max_lag_sizes=None,
        n_ex_ds=None,
        en_nlag=None,
        ex_nlag=None,
        n_estimators=None,
        criterion='mse',
        max_features=None,
        max_depth=None,
        min_samples_split=None,
        min_samples_leaf=None,
        bootstrap=None,
        oob_score=False,
        n_jobs=1,
        random_state=None,
        verbose=False):
    '''
    Return a pyll graph with hyperparamters that will construct
    a lagselectors.RFRLagSelector (random forest regression lag selector) 
    model.

    See help(hpsklearn.components._lags_hp_space) for details on 
    specifying lag sizes.
    '''
    def _name(msg):
        return '%s.%s_%s' % (name, 'rfr', msg)
    # Lag selector hyperparameters.
    hp_space = _lags_hp_space(_name, 
                              max_lag_sizes=max_lag_sizes, 
                              n_ex_ds=n_ex_ds,
                              en_nlag=en_nlag,
                              ex_nlag=ex_nlag)
    # Random forest hyperparameters.
    hp_space.update(_trees_hp_space(_name,
                                    n_estimators=n_estimators,
                                    criterion=criterion,
                                    max_features=max_features,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    bootstrap=bootstrap,
                                    oob_score=oob_score,
                                    n_jobs=n_jobs,
                                    random_state=random_state,
                                    verbose=verbose))
    return scope.sklearn_RFRLagSelector(**hp_space)

def etr_lags(
        name,
        max_lag_sizes=None,
        n_ex_ds=None,
        en_nlag=None,
        ex_nlag=None,
        n_estimators=None,
        criterion='mse',
        max_features=None,
        max_depth=None,
        min_samples_split=None,
        min_samples_leaf=None,
        bootstrap=None,
        oob_score=False,
        n_jobs=1,
        random_state=None,
        verbose=False):
    '''
    Return a pyll graph with hyperparamters that will construct
    a lagselectors.ETRLagSelector (extra trees regression lag selector) 
    model.

    See help(hpsklearn.components._lags_hp_space) for details on 
    specifying lag sizes.
    '''
    def _name(msg):
        return '%s.%s_%s' % (name, 'etr', msg)
    # Lag selector hyperparameters.
    hp_space = _lags_hp_space(_name, 
                              max_lag_sizes=max_lag_sizes, 
                              n_ex_ds=n_ex_ds,
                              en_nlag=en_nlag,
                              ex_nlag=ex_nlag)
    # Extra trees hyperparameters.
    hp_space.update(_trees_hp_space(_name,
                                    n_estimators=n_estimators,
                                    criterion=criterion,
                                    max_features=max_features,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    bootstrap=bootstrap,
                                    oob_score=oob_score,
                                    n_jobs=n_jobs,
                                    random_state=random_state,
                                    verbose=verbose))
    return scope.sklearn_ETRLagSelector(**hp_space)

# TODO: Pick reasonable default values
def random_forest(name,
                  n_estimators=50,
                  criterion=None,
                  max_features=None,
                  max_depth=None,
                  min_samples_split=None,
                  min_samples_leaf=None,
                  bootstrap=None,
                  oob_score=False,
                  n_jobs=1,
                  random_state=None,
                  verbose=False):

    def _name(msg):
        return '%s.%s_%s' % (name, 'rfc', msg)

    rval = scope.sklearn_RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=(_trees_criterion(_name('criterion'))
                   if criterion is None else criterion),
        max_features=(_trees_max_features(_name('max_features'))
                      if max_features is None else max_features),
        max_depth=(_trees_max_depth(_name('max_depth'))
                   if max_depth is None else max_depth),
        min_samples_split=(_trees_min_samples_split(_name('min_samples_split'))
                           if min_samples_split is None else min_samples_split),
        min_samples_leaf=(_trees_min_samples_leaf(_name('min_samples_leaf'))
                          if min_samples_leaf is None else min_samples_leaf),
        bootstrap=(_trees_bootstrap(_name('bootstrap'))
                   if bootstrap is None else bootstrap),
        oob_score=oob_score,
        n_jobs=n_jobs,
        random_state=_random_state(_name('rstate'), random_state),
        verbose=verbose,
    )
    return rval

# TODO: Pick reasonable default values


def random_forest_regression(name,
                             n_estimators=50,
                             criterion='mse',
                             max_features=None,
                             max_depth=None,
                             min_samples_split=None,
                             min_samples_leaf=None,
                             bootstrap=None,
                             oob_score=False,
                             n_jobs=1,
                             random_state=None,
                             verbose=False):

    def _name(msg):
        return '%s.%s_%s' % (name, 'rfr', msg)

    rval = scope.sklearn_RandomForestRegressor(
        n_estimators=n_estimators,
        criterion=criterion,
        max_features=(_trees_max_features(_name('max_features'))
                      if max_features is None else max_features),
        max_depth=(_trees_max_depth(_name('max_depth'))
                   if max_depth is None else max_depth),
        min_samples_split=(_trees_min_samples_split(_name('min_samples_split'))
                           if min_samples_split is None else min_samples_split),
        min_samples_leaf=(_trees_min_samples_leaf(_name('min_samples_leaf'))
                          if min_samples_leaf is None else min_samples_leaf),
        bootstrap=(_trees_bootstrap(_name('bootstrap'))
                   if bootstrap is None else bootstrap),
        oob_score=oob_score,
        n_jobs=n_jobs,
        random_state=_random_state(_name('rstate'), random_state),
        verbose=verbose,
    )
    return rval


# TODO: Pick reasonable default values
# TODO: the parameters are the same as RandomForest, stick em together somehow
def extra_trees(name,
                n_estimators=50,
                criterion=None,
                max_features=None,
                max_depth=None,
                min_samples_split=None,
                min_samples_leaf=None,
                bootstrap=None,
                oob_score=False,
                n_jobs=1,
                random_state=None,
                verbose=False):

    def _name(msg):
        return '%s.%s_%s' % (name, 'etc', msg)

    rval = scope.sklearn_ExtraTreesClassifier(
        n_estimators=n_estimators,
        criterion=(_trees_criterion(_name('criterion'))
                   if criterion is None else criterion),
        max_features=(_trees_max_features(_name('max_features'))
                      if max_features is None else max_features),
        max_depth=(_trees_max_depth(_name('max_depth'))
                   if max_depth is None else max_depth),
        min_samples_split=(_trees_min_samples_split(_name('min_samples_split'))
                           if min_samples_split is None else min_samples_split),
        min_samples_leaf=(_trees_min_samples_leaf(_name('min_samples_leaf'))
                          if min_samples_leaf is None else min_samples_leaf),
        bootstrap=(_trees_bootstrap(_name('bootstrap'))
                   if bootstrap is None else bootstrap),
        oob_score=oob_score,
        n_jobs=n_jobs,
        random_state=_random_state(_name('rstate'), random_state),
        verbose=verbose,
    )
    return rval


# TODO: Pick reasonable default values
# TODO: the parameters are the same as RandomForest, stick em together somehow
def extra_trees_regression(name,
                           n_estimators=50,
                           criterion='mse',
                           max_features=None,
                           max_depth=None,
                           min_samples_split=None,
                           min_samples_leaf=None,
                           bootstrap=None,
                           oob_score=False,
                           n_jobs=1,
                           random_state=None,
                           verbose=False):

    def _name(msg):
        return '%s.%s_%s' % (name, 'etr', msg)

    rval = scope.sklearn_ExtraTreesRegressor(
        n_estimators=n_estimators,
        criterion=criterion,
        max_features=(_trees_max_features(_name('max_features'))
                      if max_features is None else max_features),
        max_depth=(_trees_max_depth(_name('max_depth'))
                   if max_depth is None else max_depth),
        min_samples_split=(_trees_min_samples_split(_name('min_samples_split'))
                           if min_samples_split is None else min_samples_split),
        min_samples_leaf=(_trees_min_samples_leaf(_name('min_samples_leaf'))
                          if min_samples_leaf is None else min_samples_leaf),
        bootstrap=(_trees_bootstrap(_name('bootstrap'))
                   if bootstrap is None else bootstrap),
        oob_score=oob_score,
        n_jobs=n_jobs,
        random_state=_random_state(_name('rstate'), random_state),
        verbose=verbose,
    )
    return rval


def sgd(name,
        loss=None,  # default - 'hinge'
        penalty=None,  # default - 'l2'
        alpha=None,  # default - 0.0001
        l1_ratio=None,  # default - 0.15, must be within [0, 1]
        fit_intercept=True,  # default - True
        n_iter=5,  # default - 5
        shuffle=True,  # default - True
        random_state=None,  # default - None
        epsilon=None,
        n_jobs=1,  # default - 1 (-1 means all CPUs)
        learning_rate=None,  # default - 'optimal'
        eta0=None,  # default - 0.0
        power_t=None,  # default - 0.5
        class_weight='choose',
        warm_start=False,
        verbose=False):

    def _name(msg):
        return '%s.%s_%s' % (name, 'sgdc', msg)

    rval = scope.sklearn_SGDClassifier(
        loss=hp.pchoice(_name('loss'), [
            (0.25, 'hinge'),
            (0.25, 'log'),
            (0.25, 'modified_huber'),
            (0.05, 'squared_hinge'),
            (0.05, 'perceptron'),
            (0.05, 'squared_loss'),
            (0.05, 'huber'),
            (0.03, 'epsilon_insensitive'),
            (0.02, 'squared_epsilon_insensitive')
        ]) if loss is None else loss,
        penalty=_sgd_penalty(_name('penalty')) if penalty is None else penalty,
        alpha=_sgd_alpha(_name('alpha')) if alpha is None else alpha,
        l1_ratio=(_sgd_l1_ratio(_name('l1ratio'))
                  if l1_ratio is None else l1_ratio),
        fit_intercept=fit_intercept,
        n_iter=n_iter,
        learning_rate=(_sgdc_learning_rate(_name('learning_rate'))
                       if learning_rate is None else learning_rate),
        eta0=_sgd_eta0(_name('eta0')) if eta0 is None else eta0,
        power_t=_sgd_power_t(_name('power_t')) if power_t is None else power_t,
        class_weight=(_class_weight(_name('clsweight'))
                      if class_weight == 'choose' else class_weight),
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=_random_state(_name('rstate'), random_state),
    )
    return rval


def sgd_regression(name,
                   loss=None,  # default - 'squared_loss'
                   penalty=None,  # default - 'l2'
                   alpha=None,  # default - 0.0001
                   l1_ratio=None,  # default - 0.15, must be within [0, 1]
                   fit_intercept=True,  # default - True
                   n_iter=5,  # default - 5
                   shuffle=None,  # default - False
                   random_state=None,  # default - None
                   epsilon=None,  # default - 0.1
                   learning_rate=None,  # default - 'invscaling'
                   eta0=None,  # default - 0.01
                   power_t=None,  # default - 0.5
                   warm_start=False,
                   verbose=False):

    def _name(msg):
        return '%s.%s_%s' % (name, 'sgdr', msg)

    rval = scope.sklearn_SGDRegressor(
        loss=hp.pchoice(_name('loss'), [
            (0.25, 'squared_loss'),
            (0.25, 'huber'),
            (0.25, 'epsilon_insensitive'),
            (0.25, 'squared_epsilon_insensitive')
        ]) if loss is None else loss,
        penalty=_sgd_penalty(_name('penalty')) if penalty is None else penalty,
        alpha=_sgd_alpha(_name('alpha')) if alpha is None else alpha,
        l1_ratio=(_sgd_l1_ratio(_name('l1ratio'))
                  if l1_ratio is None else l1_ratio),
        fit_intercept=fit_intercept,
        n_iter=n_iter,
        # For regression, use the SVM epsilon instead of the SGD one.
        epsilon=_svm_epsilon(_name('epsilon')) if epsilon is None else epsilon,
        learning_rate=(_sgdr_learning_rate(_name('learning_rate'))
                       if learning_rate is None else learning_rate),
        eta0=_sgd_eta0(_name('eta0')) if eta0 is None else eta0,
        power_t=_sgd_power_t(_name('power_t')) if power_t is None else power_t,
        verbose=verbose,
        random_state=_random_state(_name('rstate'), random_state),
    )
    return rval

# def ridge(name,
#     alpha=None,           #default - 1.0
#     normalize=None,       #default - False,
#     tol=None,             #default - 0.001
#     solver=None,          #default - 'auto'
#     fit_intercept=None,   #default - True
#     ):

#     def _name(msg):
#       return '%s.%s_%s' % (name, 'sgd', msg)

#     rval = scope.sklearn_Ridge(
#         alpha=hp.loguniform(
#             _name('alpha'),
#             np.log(1e-3),
#             np.log(1e3)) if alpha is None else alpha,
#         normalize=hp.pchoice(
#             _name('normalize'),
#             [ (0.8, True), (0.2, False) ]) if normalize is None else normalize,
#         fit_intercept=hp.pchoice(
#             _name('fit_intercept'),
#             [ (0.8, True), (0.2, False) ]) if fit_intercept is None else fit_intercept,
#         tol=0.001 if tol is None else tol,
#         solver="auto" if solver is None else solver,
#         )
#     return rval


def multinomial_nb(name,
                   alpha=None,
                   fit_prior=None,
                   class_prior=None,
                   ):

    def _name(msg):
        return '%s.%s_%s' % (name, 'multinomial_nb', msg)

    rval = scope.sklearn_MultinomialNB(
        alpha=(hp.quniform(_name('alpha'), 0, 1, 0.001)
               if alpha is None else alpha),
        fit_prior=(hp_bool(_name('fit_prior'))
                   if fit_prior is None else fit_prior),
        class_prior=class_prior
    )
    return rval


def any_classifier(name):
    return hp.choice('%s' % name, [
        svc(name + '.svc'),
        knn(name + '.knn'),
        random_forest(name + '.random_forest'),
        extra_trees(name + '.extra_trees'),
        sgd(name + '.sgd'),
    ])


def any_sparse_classifier(name):
    return hp.choice('%s' % name, [
        liblinear_svc(name + '.linear_svc'),
        sgd(name + '.sgd'),
        knn(name + '.knn', sparse_data=True),
        multinomial_nb(name + '.multinomial_nb')
    ])


def any_regressor(name):
    return hp.choice('%s' % name, [
        svr(name + '.svr'),
        knn_regression(name + '.knn'),
        random_forest_regression(name + '.random_forest'),
        extra_trees_regression(name + '.extra_trees'),
        sgd_regression(name + '.sgd'),
    ])


def any_sparse_regressor(name):
    return hp.choice('%s' % name, [
        sgd_regression(name + '.sgd'),
        knn_regression(name + '.knn', sparse_data=True),
    ])


def pca(name, n_components=None, whiten=None, copy=True):
    rval = scope.sklearn_PCA(
        # -- qloguniform is missing a "scale" parameter so we
        #    lower the "high" parameter and multiply by 4 out front
        # n_components=4 * scope.int(
        #     hp.qloguniform(
        #         name + '.n_components',
        #         low=np.log(0.51),
        #         high=np.log(30.5),
        #         q=1.0)) if n_components is None else n_components,
        n_components=(hp.uniform(name + '.n_components', 0, 1) 
                      if n_components is None else n_components),
        whiten=hp_bool(name + '.whiten') if whiten is None else whiten,
        copy=copy,
    )
    return rval


def standard_scaler(name, with_mean=None, with_std=None):
    rval = scope.sklearn_StandardScaler(
        with_mean=hp_bool(
            name + '.with_mean',
        ) if with_mean is None else with_mean,
        with_std=hp_bool(
            name + '.with_std',
        ) if with_std is None else with_std,
    )
    return rval


def tfidf(name,
          analyzer=None,
          ngram_range=None,
          stop_words=None,
          lowercase=None,
          max_df=1.0,
          min_df=1,
          max_features=None,
          binary=None,
          norm=None,
          use_idf=False,
          smooth_idf=False,
          sublinear_tf=False,
          ):

    def _name(msg):
        return '%s.%s_%s' % (name, 'tfidf', msg)

    max_ngram = scope.int(hp.quniform(
        _name('max_ngram'),
        1, 4, 1))

    rval = scope.sklearn_Tfidf(
        stop_words=hp.choice(
            _name('stop_words'),
            ['english', None]) if analyzer is None else analyzer,
        lowercase=hp_bool(
            _name('lowercase'),
        ) if lowercase is None else lowercase,
        max_df=max_df,
        min_df=min_df,
        binary=hp_bool(
            _name('binary'),
        ) if binary is None else binary,
        ngram_range=(1, max_ngram) if ngram_range is None else ngram_range,
        norm=norm,
        use_idf=use_idf,
        smooth_idf=smooth_idf,
        sublinear_tf=sublinear_tf,
    )
    return rval


def min_max_scaler(name, feature_range=None, copy=True):
    if feature_range is None:
        feature_range = (
            hp.choice(name + '.feature_min', [-1.0, 0.0]), 1.0
        )
    rval = scope.sklearn_MinMaxScaler(
        feature_range=feature_range,
        copy=copy,
    )
    return rval


def normalizer(name, norm=None):
    rval = scope.sklearn_Normalizer(
        norm=(hp.choice(name + '.norm', ['l1', 'l2']) 
              if norm is None else norm),
    )
    return rval


def one_hot_encoder(name,
                    n_values=None,
                    categorical_features=None,
                    dtype=None):
    rval = scope.sklearn_OneHotEncoder(
        n_values='auto' if n_values is None else n_values,
        categorical_features=('all'
                              if categorical_features is None
                              else categorical_features),
        dtype=np.float if dtype is None else dtype,
    )
    return rval


def rbm(name,
        n_components=None,
        learning_rate=None,
        batch_size=None,
        n_iter=None,
        verbose=False,
        random_state=None):
    rval = scope.sklearn_BernoulliRBM(
        n_components=scope.int(
            hp.qloguniform(
                name + '.n_components',
                low=np.log(0.51),
                high=np.log(999.5),
                q=1.0)) if n_components is None else n_components,
        learning_rate=hp.lognormal(
            name + '.learning_rate',
            np.log(0.01),
            np.log(10),
        ) if learning_rate is None else learning_rate,
        batch_size=scope.int(
            hp.qloguniform(
                name + '.batch_size',
                np.log(1),
                np.log(100),
                q=1,
            )) if batch_size is None else batch_size,
        n_iter=scope.int(
            hp.qloguniform(
                name + '.n_iter',
                np.log(1),
                np.log(1000),  # -- max sweeps over the *whole* train set
                q=1,
            )) if n_iter is None else n_iter,
        verbose=verbose,
        random_state=_random_state(_name('rstate'), random_state),
    )
    return rval


def colkmeans(name,
              n_clusters=None,
              init=None,
              n_init=None,
              max_iter=None,
              tol=None,
              precompute_distances=True,
              verbose=0,
              random_state=None,
              copy_x=True,
              n_jobs=1):
    rval = scope.sklearn_ColumnKMeans(
        n_clusters=scope.int(
            hp.qloguniform(
                name + '.n_clusters',
                low=np.log(1.51),
                high=np.log(19.5),
                q=1.0)) if n_clusters is None else n_clusters,
        init=hp.choice(
            name + '.init',
            ['k-means++', 'random'],
        ) if init is None else init,
        n_init=hp.choice(
            name + '.n_init',
            [1, 2, 10, 20],
        ) if n_init is None else n_init,
        max_iter=scope.int(
            hp.qlognormal(
                name + '.max_iter',
                np.log(300),
                np.log(10),
                q=1,
            )) if max_iter is None else max_iter,
        tol=hp.lognormal(
            name + '.tol',
            np.log(0.0001),
            np.log(10),
        ) if tol is None else tol,
        precompute_distances=precompute_distances,
        verbose=verbose,
        random_state=random_state,
        copy_x=copy_x,
        n_jobs=n_jobs,
    )
    return rval

# XXX: todo GaussianRandomProjection
# XXX: todo SparseRandomProjection


def any_preprocessing(name):
    """Generic pre-processing appropriate for a wide variety of data
    """
    return hp.choice('%s' % name, [
        [pca(name + '.pca')],
        [standard_scaler(name + '.standard_scaler')],
        [min_max_scaler(name + '.min_max_scaler')],
        [normalizer(name + '.normalizer')],
        # -- not putting in one-hot because it can make vectors huge
        #[one_hot_encoder(name + '.one_hot_encoder')],
        []
    ])


def any_text_preprocessing(name):
    """Generic pre-processing appropriate for text data
    """
    return hp.choice('%s' % name, [
        [tfidf(name + '.tfidf')],
    ])


def generic_space(name='space'):
    model = hp.pchoice('%s' % name, [
        (.8, {'preprocessing': [pca(name + '.pca')],
              'classifier': any_classifier(name + '.pca_clsf')
              }),
        (.2, {'preprocessing': [min_max_scaler(name + '.min_max_scaler')],
              'classifier': any_classifier(name + '.min_max_clsf'),
              }),
    ])
    return as_apply({'model': model})

# -- flake8 eof

import numpy as np
import sklearn.svm
import sklearn.ensemble
import sklearn.tree
import sklearn.neighbors
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.neural_network
import sklearn.linear_model
import sklearn.discriminant_analysis
import sklearn.feature_extraction.text
import sklearn.naive_bayes
from functools import partial
from hyperopt.pyll import scope, as_apply
from hyperopt import hp
from .vkmeans import ColumnKMeans
from . import lagselectors


##########################################
##==== Wrappers for sklearn modules ====##
##########################################
@scope.define
def sklearn_SVC(*args, **kwargs):
    return sklearn.svm.SVC(*args, **kwargs)

@scope.define
def sklearn_SVR(*args, **kwargs):
    return sklearn.svm.SVR(*args, **kwargs)

@scope.define
def ts_LagSelector(*args, **kwargs):
    return lagselectors.LagSelector(*args, **kwargs)

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
def sklearn_AdaBoostClassifier(*args, **kwargs):
    return sklearn.ensemble.AdaBoostClassifier(*args, **kwargs)

@scope.define
def sklearn_AdaBoostRegressor(*args, **kwargs):
    return sklearn.ensemble.AdaBoostRegressor(*args, **kwargs)

@scope.define
def sklearn_GradientBoostingClassifier(*args, **kwargs):
    return sklearn.ensemble.GradientBoostingClassifier(*args, **kwargs)

@scope.define
def sklearn_GradientBoostingRegressor(*args, **kwargs):
    return sklearn.ensemble.GradientBoostingRegressor(*args, **kwargs)

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
def sklearn_DecisionTreeClassifier(*args, **kwargs):
    return sklearn.tree.DecisionTreeClassifier(*args, **kwargs)


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
def sklearn_PassiveAggressiveClassifier(*args, **kwargs):
    return sklearn.linear_model.PassiveAggressiveClassifier(*args, **kwargs)


@scope.define
def sklearn_LinearDiscriminantAnalysis(*args, **kwargs):
    return sklearn.discriminant_analysis.LinearDiscriminantAnalysis(*args, **kwargs)


@scope.define
def sklearn_QuadraticDiscriminantAnalysis(*args, **kwargs):
    return sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis(*args, **kwargs)


@scope.define
def sklearn_MultinomialNB(*args, **kwargs):
    return sklearn.naive_bayes.MultinomialNB(*args, **kwargs)

@scope.define
def sklearn_GaussianNB(*args, **kwargs):
    return sklearn.naive_bayes.GaussianNB(*args, **kwargs)


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


##############################
##==== Global variables ====##
##############################
_svm_default_cache_size = 512


###############################################
##==== Various hyperparameter generators ====##
###############################################
def hp_bool(name):
    return hp.choice(name, [False, True])

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

def _boosting_n_estimators(name):
    return scope.int(hp.qloguniform(name, np.log(10.5), np.log(1000.5), 1))

def _ada_boost_learning_rate(name):
    return hp.lognormal(name, np.log(0.01), np.log(10.0))

def _ada_boost_loss(name):
    return hp.choice(name, ['linear', 'square', 'exponential'])

def _ada_boost_algo(name):
    return hp.choice(name, ['SAMME', 'SAMME.R'])

def _grad_boosting_reg_loss_alpha(name):
    return hp.choice(name, [
        ('ls', 0.9), 
        ('lad', 0.9), 
        ('huber', hp.uniform(name + '.alpha', 0.85, 0.95)), 
        ('quantile', 0.5)
    ])

def _grad_boosting_clf_loss(name):
    return hp.choice(name, ['deviance', 'exponential'])

def _grad_boosting_learning_rate(name):
    return hp.lognormal(name, np.log(0.01), np.log(10.0))

def _grad_boosting_subsample(name):
    return hp.pchoice(name, [
        (0.2, 1.0),  # default choice.
        (0.8, hp.uniform(name + '.sgb', 0.5, 1.0))  # stochastic grad boosting.
    ])

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


##############################################
##==== SVM hyperparameters search space ====##
##############################################
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
        gamma_ /= n_features  # make gamma independent of n_features.
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


def _svc_hp_space(name_func, random_state=None, probability=False):
    '''Generate SVC specific hyperparamters
    '''
    hp_space = dict(
        random_state = _random_state(name_func('rstate'),random_state),
        probability=probability
    )
    return hp_space

def _svr_hp_space(name_func, epsilon=None):
    '''Generate SVR specific hyperparamters
    '''
    hp_space = {}
    hp_space['epsilon'] = (_svm_epsilon(name_func('epsilon')) 
                           if epsilon is None else epsilon)
    return hp_space


#########################################
##==== SVM classifier constructors ====##
#########################################
def svc_kernel(name, kernel, random_state=None, probability=False, **kwargs):
    """
    Return a pyll graph with hyperparamters that will construct
    a sklearn.svm.SVC model with a user specified kernel.

    See help(hpsklearn.components._svm_hp_space) for info on additional SVM
    arguments.
    """
    def _name(msg):
        return '%s.%s_%s' % (name, kernel, msg)

    hp_space = _svm_hp_space(_name, kernel=kernel, **kwargs)
    hp_space.update(_svc_hp_space(_name, random_state, probability))
    return scope.sklearn_SVC(**hp_space)

def svc_linear(name, **kwargs):
    '''Simply use the svc_kernel function with kernel fixed as linear to 
    return an SVC object.
    '''
    return svc_kernel(name, kernel='linear', **kwargs)

def svc_rbf(name, **kwargs):
    '''Simply use the svc_kernel function with kernel fixed as rbf to 
    return an SVC object.
    '''
    return svc_kernel(name, kernel='rbf', **kwargs)

def svc_poly(name, **kwargs):
    '''Simply use the svc_kernel function with kernel fixed as poly to 
    return an SVC object.
    '''
    return svc_kernel(name, kernel='poly', **kwargs)

def svc_sigmoid(name, **kwargs):
    '''Simply use the svc_kernel function with kernel fixed as sigmoid to 
    return an SVC object.
    '''
    return svc_kernel(name, kernel='sigmoid', **kwargs)

def svc(name, kernels=['linear', 'rbf', 'poly', 'sigmoid'], **kwargs):
    svms = {
        'linear': partial(svc_linear, name=name),
        'rbf': partial(svc_rbf, name=name),
        'poly': partial(svc_poly, name=name),
        'sigmoid': partial(svc_sigmoid, name=name),
    }
    choices = [svms[kern](**kwargs) for kern in kernels]
    if len(choices) == 1:
        rval = choices[0]
    else:
        rval = hp.choice('%s.kernel' % name, choices)
    return rval

########################################
##==== SVM regressor constructors ====##
########################################
def svr_kernel(name, kernel, epsilon=None, **kwargs):
    """
    Return a pyll graph with hyperparamters that will construct
    a sklearn.svm.SVR model with a user specified kernel.

    Args:
        epsilon([float]): tolerance on regression errors.

    See help(hpsklearn.components._svm_hp_space) for info on additional SVM
    arguments.
    """
    def _name(msg):
        return '%s.%s_%s' % (name, kernel, msg)

    hp_space = _svm_hp_space(_name, kernel=kernel, **kwargs)
    hp_space.update(_svr_hp_space(_name, epsilon))
    return scope.sklearn_SVR(**hp_space)

def svr_linear(name, **kwargs):
    '''Simply use the svr_kernel function with kernel fixed as linear to 
    return an SVR object.
    '''
    return svr_kernel(name, kernel='linear', **kwargs)

def svr_rbf(name, **kwargs):
    '''Simply use the svr_kernel function with kernel fixed as rbf to 
    return an SVR object.
    '''
    return svr_kernel(name, kernel='rbf', **kwargs)

def svr_poly(name, **kwargs):
    '''Simply use the svr_kernel function with kernel fixed as poly to 
    return an SVR object.
    '''
    return svr_kernel(name, kernel='poly', **kwargs)

def svr_sigmoid(name, **kwargs):
    '''Simply use the svr_kernel function with kernel fixed as sigmoid to 
    return an SVR object.
    '''
    return svr_kernel(name, kernel='sigmoid', **kwargs)

def svr(name, kernels=['linear', 'rbf', 'poly', 'sigmoid'], **kwargs):
    svms = {
        'linear': partial(svr_linear, name=name),
        'rbf': partial(svr_rbf, name=name),
        'poly': partial(svr_poly, name=name),
        'sigmoid': partial(svr_sigmoid, name=name),
    }
    choices = [svms[kern](**kwargs) for kern in kernels]
    if len(choices) == 1:
        rval = choices[0]
    else:
        rval = hp.choice('%s.kernel' % name, choices)
    return rval

##################################################
##==== Liblinear SVM classifier constructor ====##
##################################################
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


##############################################
##==== KNN hyperparameters search space ====##
##############################################
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
        metric=metric_p[0] if metric is None else metric,
        p=metric_p[1] if p is None else p,
        metric_params=metric_params,
        n_jobs=n_jobs)
    return hp_space

###################################################
##==== KNN classifier/regressor constructors ====##
###################################################
def knn(name, **kwargs):
    '''
    Return a pyll graph with hyperparamters that will construct
    a sklearn.neighbors.KNeighborsClassifier model.
    
    See help(hpsklearn.components._knn_hp_space) for info on available KNN 
    arguments.    
    '''
    def _name(msg):
        return '%s.%s_%s' % (name, 'knc', msg)

    hp_space = _knn_hp_space(_name, **kwargs)
    return scope.sklearn_KNeighborsClassifier(**hp_space)


def knn_regression(name, **kwargs):
    '''
    Return a pyll graph with hyperparamters that will construct
    a sklearn.neighbors.KNeighborsRegressor model.
    
    See help(hpsklearn.components._knn_hp_space) for info on available KNN 
    arguments.    
    '''
    def _name(msg):
        return '%s.%s_%s' % (name, 'knr', msg)

    hp_space = _knn_hp_space(_name, **kwargs)
    return scope.sklearn_KNeighborsRegressor(**hp_space)


####################################################################
##==== Random forest/extra trees hyperparameters search space ====##
####################################################################
def _trees_hp_space(
        name_func,
        n_estimators=None,
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

#############################################################
##==== Random forest classifier/regressor constructors ====##
#############################################################
def random_forest(name, criterion=None, **kwargs):
    '''
    Return a pyll graph with hyperparamters that will construct
    a sklearn.ensemble.RandomForestClassifier model.

    Args:
        criterion([str]): choose 'gini' or 'entropy'.
    
    See help(hpsklearn.components._trees_hp_space) for info on additional 
    available random forest/extra trees arguments.    
    '''
    def _name(msg):
        return '%s.%s_%s' % (name, 'rfc', msg)

    hp_space = _trees_hp_space(_name, **kwargs)
    hp_space['criterion'] = (_trees_criterion(_name('criterion'))
                             if criterion is None else criterion)
    return scope.sklearn_RandomForestClassifier(**hp_space)


def random_forest_regression(name, criterion='mse', **kwargs):
    '''
    Return a pyll graph with hyperparamters that will construct
    a sklearn.ensemble.RandomForestRegressor model.

    Args:
        criterion([str]): 'mse' is the only choice.
    
    See help(hpsklearn.components._trees_hp_space) for info on additional 
    available random forest/extra trees arguments.    
    '''
    def _name(msg):
        return '%s.%s_%s' % (name, 'rfr', msg)

    hp_space = _trees_hp_space(_name, **kwargs)
    hp_space['criterion'] = criterion
    return scope.sklearn_RandomForestRegressor(**hp_space)


###################################################
##==== AdaBoost hyperparameters search space ====##
###################################################
def _ada_boost_hp_space(
    name_func,
    base_estimator=None,
    n_estimators=None,
    learning_rate=None,
    random_state=None):
    '''Generate AdaBoost hyperparameters search space
    '''
    hp_space = dict(
        base_estimator=base_estimator,
        n_estimators=(_boosting_n_estimators(name_func('n_estimators')) 
                      if n_estimators is None else n_estimators),
        learning_rate=(_ada_boost_learning_rate(name_func('learning_rate')) 
                       if learning_rate is None else learning_rate),
        random_state=_random_state(name_func('rstate'), random_state) 
    )
    return hp_space


########################################################
##==== AdaBoost classifier/regressor constructors ====##
########################################################
def ada_boost(name, algorithm=None, **kwargs):
    '''
    Return a pyll graph with hyperparamters that will construct
    a sklearn.ensemble.AdaBoostClassifier model.

    Args:
        algorithm([str]): choose from ['SAMME', 'SAMME.R']
    
    See help(hpsklearn.components._ada_boost_hp_space) for info on 
    additional available AdaBoost arguments.    
    '''
    def _name(msg):
        return '%s.%s_%s' % (name, 'ada_boost', msg)

    hp_space = _ada_boost_hp_space(_name, **kwargs)
    hp_space['algorithm'] = (_ada_boost_algo(_name('algo')) if 
                             algorithm is None else algorithm)
    return scope.sklearn_AdaBoostClassifier(**hp_space)


def ada_boost_regression(name, loss=None, **kwargs):
    '''
    Return a pyll graph with hyperparamters that will construct
    a sklearn.ensemble.AdaBoostRegressor model.

    Args:
        loss([str]): choose from ['linear', 'square', 'exponential']
    
    See help(hpsklearn.components._ada_boost_hp_space) for info on 
    additional available AdaBoost arguments.    
    '''
    def _name(msg):
        return '%s.%s_%s' % (name, 'ada_boost_reg', msg)

    hp_space = _ada_boost_hp_space(_name, **kwargs)
    hp_space['loss'] = (_ada_boost_loss(_name('loss')) if 
                        loss is None else loss)
    return scope.sklearn_AdaBoostRegressor(**hp_space)


###########################################################
##==== GradientBoosting hyperparameters search space ====##
###########################################################
def _grad_boosting_hp_space(
    name_func,
    learning_rate=None, 
    n_estimators=None, 
    subsample=None, 
    min_samples_split=None, 
    min_samples_leaf=None, 
    max_depth=None, 
    init=None, 
    random_state=None, 
    max_features=None, 
    verbose=0, 
    max_leaf_nodes=None, 
    warm_start=False, 
    presort='auto'):
    '''Generate GradientBoosting hyperparameters search space
    '''
    hp_space = dict(
        learning_rate=(_grad_boosting_learning_rate(name_func('learning_rate')) 
                       if learning_rate is None else learning_rate),
        n_estimators=(_boosting_n_estimators(name_func('n_estimators')) 
                      if n_estimators is None else n_estimators),
        subsample=(_grad_boosting_subsample(name_func('subsample')) 
                   if subsample is None else subsample),
        min_samples_split=(_trees_min_samples_split(name_func('min_samples_split')) 
                           if min_samples_split is None else min_samples_split),
        min_samples_leaf=(_trees_min_samples_leaf(name_func('min_samples_leaf')) 
                          if min_samples_leaf is None else min_samples_leaf),
        max_depth=(_trees_max_depth(name_func('max_depth')) 
                   if max_depth is None else max_depth),
        init=init,
        random_state=_random_state(name_func('rstate'), random_state),
        max_features=(_trees_max_features(name_func('max_features')) 
                   if max_features is None else max_features),
        warm_start=warm_start,
        presort=presort
    )
    return hp_space


################################################################
##==== GradientBoosting classifier/regressor constructors ====##
################################################################
def gradient_boosting(name, loss=None, **kwargs):
    '''
    Return a pyll graph with hyperparamters that will construct
    a sklearn.ensemble.GradientBoostingClassifier model.

    Args:
        loss([str]): choose from ['deviance', 'exponential']
    
    See help(hpsklearn.components._grad_boosting_hp_space) for info on 
    additional available GradientBoosting arguments.    
    '''
    def _name(msg):
        return '%s.%s_%s' % (name, 'gradient_boosting', msg)

    hp_space = _grad_boosting_hp_space(_name, **kwargs)
    hp_space['loss'] = (_grad_boosting_clf_loss(_name('loss')) 
                        if loss is None else loss)
    return scope.sklearn_GradientBoostingClassifier(**hp_space)


def gradient_boosting_regression(name, loss=None, alpha=None, **kwargs):
    '''
    Return a pyll graph with hyperparamters that will construct
    a sklearn.ensemble.GradientBoostingRegressor model.

    Args:
        loss([str]): choose from ['ls', 'lad', 'huber', 'quantile']
        alpha([float]): alpha parameter for huber and quantile losses. 
                        Must be within [0.0, 1.0].
    
    See help(hpsklearn.components._grad_boosting_hp_space) for info on 
    additional available GradientBoosting arguments.    
    '''
    def _name(msg):
        return '%s.%s_%s' % (name, 'gradient_boosting_reg', msg)

    loss_alpha = _grad_boosting_reg_loss_alpha(_name('loss_alpha'))
    hp_space = _grad_boosting_hp_space(_name, **kwargs)
    hp_space['loss'] = loss_alpha[0] if loss is None else loss
    hp_space['alpha'] = loss_alpha[1] if alpha is None else alpha
    return scope.sklearn_GradientBoostingRegressor(**hp_space)


###########################################################
##==== Extra trees classifier/regressor constructors ====##
###########################################################
def extra_trees(name, criterion=None, **kwargs):
    '''
    Return a pyll graph with hyperparamters that will construct
    a sklearn.ensemble.ExtraTreesClassifier model.

    Args:
        criterion([str]): choose 'gini' or 'entropy'.
    
    See help(hpsklearn.components._trees_hp_space) for info on additional 
    available random forest/extra trees arguments.    
    '''

    def _name(msg):
        return '%s.%s_%s' % (name, 'etc', msg)

    hp_space = _trees_hp_space(_name, **kwargs)
    hp_space['criterion'] = (_trees_criterion(_name('criterion'))
                             if criterion is None else criterion)
    return scope.sklearn_ExtraTreesClassifier(**hp_space)


def extra_trees_regression(name, criterion='mse', **kwargs):
    '''
    Return a pyll graph with hyperparamters that will construct
    a sklearn.ensemble.ExtraTreesRegressor model.

    Args:
        criterion([str]): 'mse' is the only choice.
    
    See help(hpsklearn.components._trees_hp_space) for info on additional 
    available random forest/extra trees arguments.    
    '''
    def _name(msg):
        return '%s.%s_%s' % (name, 'etr', msg)

    hp_space = _trees_hp_space(_name, **kwargs)
    hp_space['criterion'] = criterion
    return scope.sklearn_ExtraTreesRegressor(**hp_space)


##################################################
##==== Decision tree classifier constructor ====##
##################################################
def decision_tree(name,
                  criterion=None,
                  splitter=None,
                  max_features=None,
                  max_depth=None,
                  min_samples_split=None,
                  min_samples_leaf=None,
                  presort=False,
                  random_state=None):

    def _name(msg):
        return '%s.%s_%s' % (name, 'sgd', msg)

    rval = scope.sklearn_DecisionTreeClassifier(
        criterion=hp.choice(
            _name('criterion'),
            ['gini', 'entropy']) if criterion is None else criterion,
        splitter=hp.choice(
            _name('splitter'),
            ['best', 'random']) if splitter is None else splitter,
        max_features=hp.choice(
            _name('max_features'),
            ['sqrt', 'log2',
             None]) if max_features is None else max_features,
        max_depth=max_depth,
        min_samples_split=hp.quniform(
            _name('min_samples_split'),
            1, 10, 1) if min_samples_split is None else min_samples_split,
        min_samples_leaf=hp.quniform(
            _name('min_samples_leaf'),
            1, 5, 1) if min_samples_leaf is None else min_samples_leaf,
        presort=presort, 
        random_state=_random_state(_name('rstate'), random_state),
        )
    return rval


###################################################
##==== SGD classifier/regressor constructors ====##
###################################################
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


#################################################
##==== Naive Bayes classifiers constructor ====##
#################################################
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

def gaussian_nb(name):
    def _name(msg):
      return '%s.%s_%s' % (name, 'gaussian_nb', msg)

    rval = scope.sklearn_GaussianNB()
    return rval


###########################################
##==== Passive-aggressive classifier ====##
###########################################
def passive_aggressive(name,
    loss=None,
    C=None,
    fit_intercept=False,
    n_iter=None,
    n_jobs=1,
    shuffle=True,
    random_state=None,
    verbose=False):

    def _name(msg):
        return '%s.%s_%s' % (name, 'sgd', msg)

    rval = scope.sklearn_PassiveAggressiveClassifier(
        loss=hp.choice(
            _name('loss'),
            ['hinge', 'squared_hinge']) if loss is None else loss,
        C=hp.lognormal(
            _name('learning_rate'),
            np.log(0.01),
            np.log(10),
            ) if C is None else C,
        fit_intercept=fit_intercept,
        n_iter=scope.int(
            hp.qloguniform(
                _name('n_iter'),
                np.log(1),
                np.log(1000),
                q=1,
                )) if n_iter is None else n_iter,
        n_jobs=n_jobs,
        random_state=_random_state(_name('rstate'), random_state),
        verbose=verbose
        )
    return rval


###############################################
##==== Discriminant analysis classifiers ====##
###############################################
def linear_discriminant_analysis(name,
    solver=None,
    shrinkage=None,
    priors=None,
    n_components=None,
    store_covariance=False,
    tol=0.00001):

    def _name(msg):
        return '%s.%s_%s' % (name, 'lda', msg)

    solver_shrinkage = hp.choice(_name('solver_shrinkage_dual'),
                                     [('svd', None),
                                      ('lsqr', None),
                                      ('lsqr', 'auto'),
                                      ('eigen', None),
                                      ('eigen', 'auto')])

    rval = scope.sklearn_LinearDiscriminantAnalysis(
        solver=solver_shrinkage[0] if solver is None else solver,
        shrinkage=solver_shrinkage[1] if shrinkage is None else shrinkage,
        priors=priors,
        n_components=4 * scope.int(
            hp.qloguniform(
                _name('n_components'),
                low=np.log(0.51),
                high=np.log(30.5),
                q=1.0)) if n_components is None else n_components,
        store_covariance=store_covariance,
        tol=tol
        )
    return rval


def quadratic_discriminant_analysis(name,
    reg_param=None,
    priors=None):

    def _name(msg):
        return '%s.%s_%s' % (name, 'qda', msg)

    rval = scope.sklearn_QuadraticDiscriminantAnalysis(
        reg_param=hp.uniform(
            _name('reg_param'),
            0.0, 1.0) if reg_param is None else 0.0,
        priors=priors
        )
    return rval


####################################################
##==== Various classifier/regressor selectors ====##
####################################################
def any_classifier(name):
    return hp.choice('%s' % name, [
        svc(name + '.svc'),
        knn(name + '.knn'),
        random_forest(name + '.random_forest'),
        extra_trees(name + '.extra_trees'),
        ada_boost(name + '.ada_boost'),
        gradient_boosting(name + '.grad_boosting', loss='deviance'),
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
        ada_boost_regression(name + '.ada_boost'),
        gradient_boosting_regression(name + '.grad_boosting'),
        sgd_regression(name + '.sgd'),
    ])


def any_sparse_regressor(name):
    return hp.choice('%s' % name, [
        sgd_regression(name + '.sgd'),
        knn_regression(name + '.knn', sparse_data=True),
    ])


###############################################
##==== Various preprocessor constructors ====##
###############################################
def pca(name, n_components=None, whiten=None, copy=True):
    rval = scope.sklearn_PCA(
        # -- qloguniform is missing a "scale" parameter so we
        #    lower the "high" parameter and multiply by 4 out front
        n_components=4 * scope.int(
            hp.qloguniform(
                name + '.n_components',
                low=np.log(0.51),
                high=np.log(30.5),
                q=1.0)) if n_components is None else n_components,
        # n_components=(hp.uniform(name + '.n_components', 0, 1) 
        #               if n_components is None else n_components),
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

def ts_lagselector(name, lower_lags=1, upper_lags=1):
    rval = scope.ts_LagSelector(
        lag_size=scope.int(
            hp.quniform(name + '.lags', 
                        lower_lags - .5, upper_lags + .5, 1))
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

    def _name(msg):
        return '%s.%s_%s' % (name, 'rbm', msg)

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


####################################
##==== Preprocessor selectors ====##
####################################
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


##############################################################
##==== Generic hyperparameters search space constructor ====##
##############################################################
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

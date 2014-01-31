import numpy as np
import sklearn.svm
import sklearn.ensemble
import sklearn.neighbors
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.neural_network
from hyperopt.pyll import scope
from hyperopt import hp
from .vkmeans import ColumnKMeans

@scope.define
def sklearn_SVC(*args, **kwargs):
    return sklearn.svm.SVC(*args, **kwargs)


@scope.define
def sklearn_LinearSVC(*args, **kwargs):
    return sklearn.svm.LinearSVC(*args, **kwargs)


@scope.define
def sklearn_KNeighborsClassifier(*args, **kwargs):
    return sklearn.neighbors.KNeighborsClassifier(*args, **kwargs)


@scope.define
def sklearn_RandomForestClassifier(*args, **kwargs):
    return sklearn.ensemble.RandomForestClassifier(*args, **kwargs)


@scope.define
def sklearn_ExtraTreesClassifier(*args, **kwargs):
    return sklearn.ensemble.ExtraTreesClassifier(*args, **kwargs)


@scope.define
def sklearn_PCA(*args, **kwargs):
    return sklearn.decomposition.PCA(*args, **kwargs)


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


def svc_linear(name,
    C=None,
    shrinking=None,
    tol=None,
    max_iter=None,
    verbose=False,
    cache_size=100.,
    ):
    """
    Return a pyll graph with hyperparamters that will construct
    a sklearn.svm.SVC model with a linear kernel.

    """
    def _name(msg):
        return '%s.%s_%s' % (name, 'linear', msg)

    rval = scope.sklearn_SVC(
        kernel='linear',
        C=hp.lognormal(
            _name('C'),
            np.log(1.0),
            np.log(4.0)) if C is None else C,
        shrinking=hp_bool(
            _name('shrinking')) if shrinking is None else shrinking,
        tol=scope.inv_patience_param(
            hp.lognormal(
                _name('tol'),
                np.log(1e-3),
                np.log(10))) if tol is None else tol,
        # -- this is basically only here to prevent
        #    an infinite loop in libsvm's solver.
        #    A more useful control mechanism might be a timer
        #    or a kill msg to a sub-process or something...
        max_iter=scope.patience_param(scope.int(
            hp.qloguniform(
                _name('max_iter'),
                np.log(1000),
                np.log(100000),
                q=10,
                ))) if max_iter is None else max_iter,
        verbose=verbose,
        cache_size=cache_size,
        )
    return rval


def svc_rbf(name,
    C=None,
    gamma=None,
    shrinking=None,
    tol=None,
    max_iter=None,
    verbose=False,
    cache_size=100.,
    ):
    """
    Return a pyll graph with hyperparamters that will construct
    a sklearn.svm.SVC model with an RBF kernel.

    """
    def _name(msg):
        return '%s.%s_%s' % (name, 'rbf', msg)

    rval = scope.sklearn_SVC(
        kernel='rbf',
        C=hp.lognormal(
            _name('C'),
            np.log(1.0),
            np.log(4.0)) if C is None else C,
        gamma=hp.lognormal(
            _name('gamma'),
            np.log(1.0),
            np.log(3.0)) if gamma is None else gamma,
        shrinking=hp_bool(
            _name('shrinking')) if shrinking is None else shrinking,
        tol=hp.lognormal(
            _name('tol'),
            np.log(1e-3),
            np.log(10)) if tol is None else tol,
        # -- this is basically only here to prevent
        #    an infinite loop in libsvm's solver.
        #    A more useful control mechanism might be a timer
        #    or a kill mesg to a sub-process or something...
        max_iter=scope.patience_param(scope.int(
            hp.qloguniform(
                _name('max_iter'),
                np.log(1000),
                np.log(100000),
                q=10,
                ))) if max_iter is None else max_iter,
        verbose=verbose,
        cache_size=cache_size,
        )
    return rval


def svc_poly(name,
    C=None,
    gamma=None,
    coef0=None,
    degree=None,
    shrinking=None,
    tol=None,
    max_iter=None,
    verbose=False,
    cache_size=100.,
    ):
    """
    Return a pyll graph with hyperparamters that will construct
    a sklearn.svm.SVC model with an RBF kernel.

    """
    def _name(msg):
        return '%s.%s_%s' % (name, 'poly', msg)

    rval = scope.sklearn_SVC(
        kernel='poly',
        C=hp.lognormal(
            _name('C'),
            np.log(1.0),
            np.log(4.0)) if C is None else C,
        gamma=hp.lognormal(
            _name('gamma'),
            np.log(1.0),
            np.log(3.0)) if gamma is None else gamma,
        coef0=hp.normal(
            _name('coef0'),
            0.0,
            1.0) if coef0 is None else coef0,
        degree=hp.quniform(
            _name('degree'),
            low=1.5,
            high=6.5,
            q=1) if degree is None else degree,
        shrinking=hp_bool(
            _name('shrinking')) if shrinking is None else shrinking,
        tol=hp.lognormal(
            _name('tol'),
            np.log(1e-3),
            np.log(10)) if tol is None else tol,
        # -- this is basically only here to prevent
        #    an infinite loop in libsvm's solver.
        #    A more useful control mechanism might be a timer
        #    or a kill mesg to a sub-process or something...
        max_iter=scope.patience_param(scope.int(
            hp.qloguniform(
                _name('max_iter'),
                np.log(1000),
                np.log(100000),
                q=10,
                ))) if max_iter is None else max_iter,
        verbose=verbose,
        cache_size=cache_size,
        )
    return rval


def svc_sigmoid(name,
    C=None,
    gamma=None,
    coef0=None,
    shrinking=None,
    tol=None,
    max_iter=None,
    verbose=False,
    cache_size=100.,
    ):
    """
    Return a pyll graph with hyperparamters that will construct
    a sklearn.svm.SVC model with an RBF kernel.

    """
    def _name(msg):
        return '%s.%s_%s' % (name, 'sigmoid', msg)

    rval = scope.sklearn_SVC(
        kernel='sigmoid',
        C=hp.lognormal(
            _name('C'),
            np.log(1.0),
            np.log(4.0)) if C is None else C,
        gamma=hp.lognormal(
            _name('gamma'),
            np.log(1.0),
            np.log(3.0)) if gamma is None else gamma,
        coef0=hp.normal(
            _name('coef0'),
            0.0,
            1.0) if coef0 is None else coef0,
        shrinking=hp_bool(
            _name('shrinking')) if shrinking is None else shrinking,
        tol=hp.lognormal(
            _name('tol'),
            np.log(1e-3),
            np.log(10)) if tol is None else tol,
        # -- this is basically only here to prevent
        #    an infinite loop in libsvm's solver.
        #    A more useful control mechanism might be a timer
        #    or a kill mesg to a sub-process or something...
        max_iter=scope.patience_param(scope.int(
            hp.qloguniform(
                _name('max_iter'),
                np.log(1000),
                np.log(100000),
                q=10,
                ))) if max_iter is None else max_iter,
        verbose=verbose,
        cache_size=cache_size,
        )
    return rval


def svc(name,
    C=None,
    kernels=['linear', 'rbf', 'poly', 'sigmoid'],
    gamma=None,
    shrinking=None,
    max_iter=None,
    verbose=False
    ):
    svms = {
        'linear': svc_linear(name,
            C=C,
            shrinking=shrinking,
            max_iter=max_iter,
            verbose=verbose),
        'rbf': svc_rbf(name,
            C=C,
            gamma=gamma,
            shrinking=shrinking,
            max_iter=max_iter,
            verbose=verbose),
        'poly': svc_poly(name,
            C=C,
            gamma=gamma,
            shrinking=shrinking,
            max_iter=max_iter,
            verbose=verbose),
        'sigmoid': svc_sigmoid(name,
            C=C,
            gamma=gamma,
            shrinking=shrinking,
            max_iter=max_iter,
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
    fit_intercept=None,
    intercept_scaling=None,
    class_weight=None,
    verbose=False,
    ):

    def _name(msg):
      return '%s.%s_%s' % (name, 'linear_svc', msg)
    
    """
    The combination of penalty='l1' and loss='l1' is not supported
    penalty='l2' and ploss='l1' is only supported when dual='true'
    penalty='l1' is only supported when dual='false'
    """
    loss_penalty_dual = hp.choice( _name('loss_penalty_dual'), 
                                  [ ('l1', 'l2', True), 
                                    ('l2', 'l2', True),
                                    ('l2', 'l1', False),
                                    ('l2', 'l2', False) ] )

    rval = scope.sklearn_LinearSVC(
        C=hp.lognormal(
            _name('C'),
            np.log(1.0),
            np.log(4.0)) if C is None else C,
        loss=loss_penalty_dual[0] if loss is None else loss,
        penalty=loss_penalty_dual[1] if penalty is None else penalty,
        dual=loss_penalty_dual[2] if dual is None else dual,
        tol=hp.lognormal(
            _name('tol'),
            np.log(1e-3),
            np.log(10)) if tol is None else tol,
        multi_class=hp.choice(
            _name('multi_class'),
            ['ovr', 'crammer_singer']) if multi_class is None else multi_class,
        fit_intercept=hp.choice(
            _name('fit_intercept'),
            [True, False]) if fit_intercept is None else fit_intercept,
        verbose=verbose,
        )
    return rval

# TODO: Pick reasonable default values
def knn(name,
    n_neighbors=None,
    weights=None,
    algorithm=None,
    leaf_size=None,
    metric=None,
    p=None,
    **kwargs
    ):

    def _name(msg):
      return '%s.%s_%s' % (name, 'knn', msg)
    
    """
    metric_arg = hp.choice( _name('metric'), [
      ( 'euclidean', None, None, None ),
      ( 'manhattan', None, None, None ),
      ( 'chebyshev', None, None, None ),
      ( 'minkowski', hp.quniform( _name('minkowski_p'), 1, 5, 1 ), None, None ),
      ( 'wminkowski', hp.quniform( _name('wminkowski_p'), 1, 5, 1 ), 
                      hp.uniform( _name('wminkowski_w'), 0, 100 ), None ),
      ( 'seuclidean', None, None, hp.uniform( _name('seuclidean_V'), 0, 100 ) ),
      ( 'mahalanobis', None, None, hp.uniform( _name('mahalanobis_V'), 0, 100 ) ),
    ] )
    """
    metric_args = hp.choice( _name('metric'), [
      { 'metric':'euclidean' },
      { 'metric':'manhattan' },
      { 'metric':'chebyshev' },
      { 'metric':'minkowski', 
        'p':scope.int(hp.quniform( _name('minkowski_p'), 1, 5, 1))},
      { 'metric':'wminkowski', 
        'p':scope.int(hp.quniform( _name('wminkowski_p'), 1, 5, 1)),
        'w':hp.uniform( _name('wminkowski_w'), 0, 100 ) },
      { 'metric':'seuclidean', 
        'V':hp.uniform( _name('seuclidean_V'), 0, 100 ) },
      { 'metric':'mahalanobis', 
        'V':hp.uniform( _name('mahalanobis_V'), 0, 100 ) },
    ] )


    rval = scope.sklearn_KNeighborsClassifier(
        n_neighbors=scope.int(hp.quniform(
            _name('n_neighbors'),
            1, 10, 1)) if n_neighbors is None else n_neighbors,
        weights=hp.choice(
            _name('weights'),
            [ 'uniform', 'distance' ] ) if weights is None else weights,
        algorithm=hp.choice(
            _name('algorithm'),
            [ 'ball_tree', 'kd_tree', 
              'brute', 'auto' ] ) if algorithm is None else algorithm,
        leaf_size=scope.int(hp.quniform(
            _name('leaf_size'),
            1, 100, 1)) if leaf_size is None else leaf_size,
        #TODO: more metrics available
        ###metric_args,
        ##metric=metric_arg[0] if metric is None else metric,
        ##p=metric_arg[1],
        ##w=metric_arg[2],
        ##V=metric_arg[3],
        #metric=hp.choice(
        #    _name('metric'),
        #    [ 'euclidean', 'manhattan', 'chebyshev', 
        #      'minkowski' ] ) if metric is None else metric,
        #p=hp.quniform(
        #    _name('p'),
        #    1, 5, 1 ) if p is None else p,
        )
    return rval

# TODO: Pick reasonable default values
def random_forest(name,
    n_estimators=None,
    criterion=None,
    max_features=None,
    max_depth=None,
    min_samples_split=None,
    min_samples_leaf=None,
    bootstrap=None,
    oob_score=None,
    n_jobs=1,
    verbose=False,
    ):

    def _name(msg):
      return '%s.%s_%s' % (name, 'random_forest', msg)
    
    """
    Out of bag estimation only available if bootstrap=True
    """

    bootstrap_oob = hp.choice( _name('bootstrap_oob'),
                              [ ( True, True ),
                                ( True, False ),
                                ( False, False ) ] )

    rval = scope.sklearn_RandomForestClassifier(
        n_estimators=scope.int( hp.quniform(
            _name('n_estimators'),
            1, 50, 1 ) ) if n_estimators is None else n_estimators,
        criterion=hp.choice(
            _name('criterion'),
            [ 'gini', 'entropy' ] ) if criterion is None else criterion,
        max_features=hp.choice(
            _name('max_features'),
            [ 'sqrt', 'log2', 
              None ] ) if max_features is None else max_features,
        max_depth=max_depth,
        min_samples_split=hp.quniform(
            _name('min_samples_split'),
            1, 10, 1 ) if min_samples_split is None else min_samples_split,
        min_samples_leaf=hp.quniform(
            _name('min_samples_leaf'),
            1, 5, 1 ) if min_samples_leaf is None else min_samples_leaf,
        bootstrap=bootstrap_oob[0] if bootstrap is None else bootstrap,
        oob_score=bootstrap_oob[1] if oob_score is None else oob_score,
        #bootstrap=hp.choice(
        #    _name('bootstrap'),
        #    [ True, False ] ) if bootstrap is None else bootstrap,
        #oob_score=hp.choice(
        #    _name('oob_score'),
        #    [ True, False ] ) if oob_score is None else oob_score,
        n_jobs=n_jobs,
        verbose=verbose,
        )
    return rval


# TODO: Pick reasonable default values
# TODO: the parameters are the same as RandomForest, stick em together somehow
def extra_trees(name,
    n_estimators=None,
    criterion=None,
    max_features=None,
    max_depth=None,
    min_samples_split=None,
    min_samples_leaf=None,
    bootstrap=None,
    oob_score=None,
    n_jobs=1,
    verbose=False,
    ):

    def _name(msg):
      return '%s.%s_%s' % (name, 'extra_trees', msg)
    
    bootstrap_oob = hp.choice( _name('bootstrap_oob'),
                              [ ( True, True ),
                                ( True, False ),
                                ( False, False ) ] )

    rval = scope.sklearn_ExtraTreesClassifier(
        n_estimators=scope.int( hp.quniform(
            _name('n_estimators'),
            1, 50, 1 ) ) if n_estimators is None else n_estimators,
        criterion=hp.choice(
            _name('criterion'),
            [ 'gini', 'entropy' ] ) if criterion is None else criterion,
        max_features=hp.choice(
            _name('max_features'),
            [ 'sqrt', 'log2', 
              None ] ) if max_features is None else max_features,
        max_depth=max_depth,
        min_samples_split=hp.quniform(
            _name('min_samples_split'),
            1, 10, 1 ) if min_samples_split is None else min_samples_split,
        min_samples_leaf=hp.quniform(
            _name('min_samples_leaf'),
            1, 5, 1 ) if min_samples_leaf is None else min_samples_leaf,
        bootstrap=bootstrap_oob[0] if bootstrap is None else bootstrap,
        oob_score=bootstrap_oob[1] if oob_score is None else oob_score,
        #bootstrap=hp.choice(
        #    _name('bootstrap'),
        #    [ True, False ] ) if bootstrap is None else bootstrap,
        #oob_score=hp.choice(
        #    _name('oob_score'),
        #    [ True, False ] ) if oob_score is None else oob_score,
        n_jobs=n_jobs,
        verbose=verbose,
        )
    return rval


def any_classifier(name):
    return hp.choice('%s' % name, [
        svc(name + '.svc'),
        liblinear_svc(name + '.linear_svc'),
        knn(name + '.knn'),
        random_forest(name + '.random_forest'),
        extra_trees(name + '.extra_trees'),
        ])

def any_sparse_classifier(name):
    return hp.choice('%s' % name, [
        svc(name + '.svc'),
        liblinear_svc(name + '.linear_svc'),
        ])


def pca(name,
    n_components=None,
    whiten=None,
    ):
    rval = scope.sklearn_PCA(
        n_components=scope.int(
            hp.qloguniform(
                name + '.n_components',
                low=np.log(0.5),
                high=np.log(999.5),
                q=1.0)) if n_components is None else n_components,
        whiten=hp_bool(
            name + '.whiten',
            ) if whiten is None else whiten,
        )
    return rval


def standard_scaler(name,
    with_mean=None,
    with_std=None,
    ):
    rval = scope.sklearn_StandardScaler(
        with_mean=hp_bool(
            name + '.with_mean',
            ) if with_mean is None else with_mean,
        with_std=hp_bool(
            name + '.with_std',
            ) if with_std is None else with_std,
        )
    return rval


def min_max_scaler(name,
    feature_range=None,
    copy=True,
    ):
    if feature_range is None:
        feature_range = (
            hp.choice(name + '.feature_min', [-1.0, 0.0]),
            1.0)
    rval = scope.sklearn_MinMaxScaler(
        feature_range=feature_range,
        copy=copy,
        )
    return rval


def normalizer(name,
    norm=None,
    ):
    rval = scope.sklearn_Normalizer(
        norm=hp.choice(
            name + '.with_mean',
            [ 'l1', 'l2' ],
            ) if norm is None else norm,
        )
    return rval


def one_hot_encoder(name,
    n_values=None,
    categorical_features=None,
    dtype=None,
    ):
    rval = scope.sklearn_OneHotEncoder(
        n_values = 'auto' if n_values is None else n_values,
        categorical_features = 'all' if categorical_features is None else
      categorical_features,
        dtype = np.float if dtype is None else dtype,
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
        random_state=random_state,
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

#XXX: todo GaussianRandomProjection
#XXX: todo SparseRandomProjection


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
    ])


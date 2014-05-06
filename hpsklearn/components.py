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
from hyperopt.pyll import scope, as_apply
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
    star_star_kwargs = kwargs.pop('starstar_kwargs')
    kwargs.update(star_star_kwargs)
    return sklearn.neighbors.KNeighborsClassifier(*args, **kwargs)


@scope.define
def sklearn_RandomForestClassifier(*args, **kwargs):
    return sklearn.ensemble.RandomForestClassifier(*args, **kwargs)


@scope.define
def sklearn_ExtraTreesClassifier(*args, **kwargs):
    return sklearn.ensemble.ExtraTreesClassifier(*args, **kwargs)


@scope.define
def sklearn_SGDClassifier(*args, **kwargs):
    return sklearn.linear_model.SGDClassifier(*args, **kwargs)


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


_svc_default_cache_size = 1000.0


def _svc_gamma(name):
    # -- making these non-conditional variables
    #    probably helps the GP algorithm generalize
    gammanz = hp.choice(name + '.gammanz', [0, 1])
    gamma = hp.lognormal(name + '.gamma', np.log(0.01), 2.5)
    return gammanz * gamma


def _svc_max_iter(name):
    return scope.patience_param(
        scope.int(
            hp.loguniform(
                name + '.max_iter',
                np.log(1e7),
                np.log(1e9))))


def _svc_C(name):
    return hp.lognormal(name + '.C', np.log(1000.0), 3.0)


def _svc_tol(name):
    return scope.inv_patience_param(
        hp.lognormal(
            name + '.tol',
            np.log(1e-3),
            2.0))

def _random_state(name, random_state):
    if random_state is None:
        return hp.randint(name, 5)
    else:
        return random_state


def svc_linear(name,
               C=None,
               shrinking=None,
               tol=None,
               max_iter=None,
               verbose=False,
               random_state=None,
               cache_size=_svc_default_cache_size):
    """
    Return a pyll graph with hyperparamters that will construct
    a sklearn.svm.SVC model with a linear kernel.

    """
    def _name(msg):
        return '%s.%s_%s' % (name, 'linear', msg)

    rval = scope.sklearn_SVC(
        kernel='linear',
        C=_svc_C(name + '.linear') if C is None else C,
        shrinking=hp_bool(
            _name('shrinking')) if shrinking is None else shrinking,
        tol=_svc_tol(name) if tol is None else tol,
        max_iter=_svc_max_iter(name) if max_iter is None else max_iter,
        verbose=verbose,
        random_state=_random_state(_name('.rstate'), random_state),
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
            random_state=None,
            cache_size=_svc_default_cache_size):
    """
    Return a pyll graph with hyperparamters that will construct
    a sklearn.svm.SVC model with an RBF kernel.

    """
    def _name(msg):
        return '%s.%s_%s' % (name, 'rbf', msg)

    rval = scope.sklearn_SVC(
        kernel='rbf',
        C=_svc_C(name + '.rbf') if C is None else C,
        gamma=_svc_gamma(name) if gamma is None else gamma,
        shrinking=hp_bool(
            _name('shrinking')) if shrinking is None else shrinking,
        tol=_svc_tol(name + '.rbf') if tol is None else tol,
        max_iter=(_svc_max_iter(name + '.rbf')
                  if max_iter is None else max_iter),
        verbose=verbose,
        cache_size=cache_size,
        random_state=_random_state(_name('rstate'), random_state),
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
             random_state=None,
             cache_size=_svc_default_cache_size):
    """
    Return a pyll graph with hyperparamters that will construct
    a sklearn.svm.SVC model with an RBF kernel.

    """
    def _name(msg):
        return '%s.%s_%s' % (name, 'poly', msg)

    # -- (K(x, y) + coef0)^d
    coef0nz = hp.choice(_name('coef0nz'), [0, 1])
    coef0 = hp.uniform(_name('coef0'), 0.0, 1.0)
    poly_coef0 = coef0nz * coef0

    rval = scope.sklearn_SVC(
        kernel='poly',
        C=_svc_C(name + '.poly') if C is None else C,
        gamma=_svc_gamma(name + '.poly') if gamma is None else gamma,
        coef0=poly_coef0 if coef0 is None else coef0,
        degree=hp.quniform(
            _name('degree'),
            low=1.5,
            high=8.5,
            q=1) if degree is None else degree,
        shrinking=hp_bool(
            _name('shrinking')) if shrinking is None else shrinking,
        tol=_svc_tol(name + '.poly') if tol is None else tol,
        max_iter=(_svc_max_iter(name + '.poly')
                  if max_iter is None else max_iter),
        verbose=verbose,
        random_state=_random_state(_name('.rstate'), random_state),
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
                random_state=None,
                cache_size=_svc_default_cache_size):
    """
    Return a pyll graph with hyperparamters that will construct
    a sklearn.svm.SVC model with an RBF kernel.

    """
    def _name(msg):
        return '%s.%s_%s' % (name, 'sigmoid', msg)

    # -- tanh(K(x, y) + coef0)
    coef0nz = hp.choice(_name('coef0nz'), [0, 1])
    coef0 = hp.normal(_name('coef0'), 0.0, 1.0)
    sigm_coef0 = coef0nz * coef0

    rval = scope.sklearn_SVC(
        kernel='sigmoid',
        C=_svc_C(name + '.sigmoid') if C is None else C,
        gamma=_svc_gamma(name + '.sigmoid') if gamma is None else gamma,
        coef0=sigm_coef0 if coef0 is None else coef0,
        shrinking=hp_bool(
            _name('shrinking')) if shrinking is None else shrinking,
        tol=_svc_tol(name + '.sigmoid') if tol is None else tol,
        max_iter=(_svc_max_iter(name + '.sigmoid')
                  if max_iter is None else max_iter),
        verbose=verbose,
        random_state=_random_state(_name('rstate'), random_state),
        cache_size=cache_size)
    return rval


def svc(name,
        C=None,
        kernels=['linear', 'rbf', 'poly', 'sigmoid'],
        shrinking=None,
        tol=None,
        max_iter=None,
        verbose=False,
        random_state=None,
        cache_size=_svc_default_cache_size):
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
            C=C,
            shrinking=shrinking,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            verbose=verbose),
        'poly': svc_poly(
            name,
            C=C,
            shrinking=shrinking,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            verbose=verbose),
        'sigmoid': svc_sigmoid(
            name,
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
                  random_state=None,
                  verbose=False):

    def _name(msg):
        return '%s.%s_%s' % (name, 'linear_svc', msg)

    """
    The combination of penalty='l1' and loss='l1' is not supported
    penalty='l2' and ploss='l1' is only supported when dual='true'
    penalty='l1' is only supported when dual='false'
    """
    loss_penalty_dual = hp.choice(_name('loss_penalty_dual'),
                                  [('l1', 'l2', True),
                                   ('l2', 'l2', True),
                                   ('l2', 'l1', False),
                                   ('l2', 'l2', False)])

    rval = scope.sklearn_LinearSVC(
        C=_svc_C(name + '.liblinear') if C is None else C,
        loss=loss_penalty_dual[0] if loss is None else loss,
        penalty=loss_penalty_dual[1] if penalty is None else penalty,
        dual=loss_penalty_dual[2] if dual is None else dual,
        tol=_svc_tol(name + '.liblinear') if tol is None else tol,
        multi_class=hp.choice(
            _name('multi_class'),
            ['ovr', 'crammer_singer']) if multi_class is None else multi_class,
        fit_intercept=hp.choice(
            _name('fit_intercept'),
            [True, False]) if fit_intercept is None else fit_intercept,
        random_state=_random_state(_name('rstate'), random_state),
        verbose=verbose,
        )
    return rval


# TODO: Pick reasonable default values
def knn(name,
        sparse_data=False,
        n_neighbors=None,
        weights=None,
        leaf_size=None,
        metric=None,
        p=None,
        **kwargs):

    def _name(msg):
        return '%s.%s_%s' % (name, 'knn', msg)

    if sparse_data:
      metric_args = { 'metric':'euclidean' }
    else:
      metric_args = hp.pchoice(_name('metric'), [
        (0.65, { 'metric':'euclidean' }),
        (0.10, { 'metric':'manhattan' }),
        (0.10, { 'metric':'chebyshev' }),
        (0.10, { 'metric':'minkowski',
          'p':scope.int(hp.quniform(_name('minkowski_p'), 1, 5, 1))}),
        (0.05, { 'metric':'wminkowski',
          'p':scope.int(hp.quniform(_name('wminkowski_p'), 1, 5, 1)),
          'w':hp.uniform( _name('wminkowski_w'), 0, 100 ) }),
      ] )

    rval = scope.sklearn_KNeighborsClassifier(
        n_neighbors=scope.int(hp.quniform(
            _name('n_neighbors'),
            0.5, 50, 1)) if n_neighbors is None else n_neighbors,
        weights=hp.choice(
            _name('weights'),
            ['uniform', 'distance']) if weights is None else weights,
        leaf_size=scope.int(hp.quniform(
            _name('leaf_size'),
            0.51, 100, 1)) if leaf_size is None else leaf_size,
        starstar_kwargs=metric_args
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
                  random_state=None,
                  verbose=False):

    def _name(msg):
        return '%s.%s_%s' % (name, 'random_forest', msg)

    """
    Out of bag estimation only available if bootstrap=True
    """

    bootstrap_oob = hp.choice(_name('bootstrap_oob'),
                              [(True, True),
                               (True, False),
                               (False, False)])

    rval = scope.sklearn_RandomForestClassifier(
        n_estimators=scope.int(hp.quniform(
            _name('n_estimators'),
            1, 50, 1)) if n_estimators is None else n_estimators,
        criterion=hp.choice(
            _name('criterion'),
            ['gini', 'entropy']) if criterion is None else criterion,
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
        bootstrap=bootstrap_oob[0] if bootstrap is None else bootstrap,
        oob_score=bootstrap_oob[1] if oob_score is None else oob_score,
        n_jobs=n_jobs,
        random_state=_random_state(_name('rstate'), random_state),
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
                random_state=None,
                verbose=False):

    def _name(msg):
        return '%s.%s_%s' % (name, 'extra_trees', msg)

    bootstrap_oob = hp.choice(_name('bootstrap_oob'),
                              [(True, True),
                               (True, False),
                               (False, False)])

    rval = scope.sklearn_ExtraTreesClassifier(
        n_estimators=scope.int(hp.quniform(
            _name('n_estimators'),
            1, 50, 1)) if n_estimators is None else n_estimators,
        criterion=hp.choice(
            _name('criterion'),
            ['gini', 'entropy']) if criterion is None else criterion,
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
        bootstrap=bootstrap_oob[0] if bootstrap is None else bootstrap,
        oob_score=bootstrap_oob[1] if oob_score is None else oob_score,
        n_jobs=n_jobs,
        random_state=_random_state(_name('rstate'), random_state),
        verbose=verbose,
        )
    return rval

def sgd(name,
    loss=None,            #default - 'hinge'
    penalty=None,         #default - 'l2'
    alpha=None,           #default - 0.0001
    l1_ratio=None,        #default - 0.15, must be within [0, 1]
    fit_intercept=None,   #default - True
    n_iter=None,          #default - 5
    shuffle=None,         #default - False
    random_state=None,    #default - None
    epsilon=None,
    n_jobs=1,             #default - 1 (-1 means all CPUs)
    learning_rate=None,   #default - 'invscaling'
    eta0=None,            #default - 0.01
    power_t=None,         #default - 0.5
    class_weight=None,
    warm_start=False,
    verbose=False,
    ):

    def _name(msg):
      return '%s.%s_%s' % (name, 'sgd', msg)
    
    rval = scope.sklearn_SGDClassifier(
        loss=hp.pchoice(
            _name('loss'),
            [ (0.25, 'hinge'), 
              (0.25, 'log'), 
              (0.25, 'modified_huber'), 
              (0.05, 'squared_hinge'), 
              (0.05, 'perceptron'),
              (0.05, 'squared_loss'), 
              (0.05, 'huber'), 
              (0.03, 'epsilon_insensitive'),
              (0.02, 'squared_epsilon_insensitive') ] ) if loss is None else loss,
        penalty=hp.pchoice(
            _name('penalty'),
            [ (0.40, 'l2'), 
              (0.35, 'l1'),
              (0.25, 'elasticnet') ] ) if penalty is None else penalty,
        alpha=hp.loguniform(
            _name('alpha'),
            np.log(1e-7),
            np.log(1)) if alpha is None else alpha,
        l1_ratio=hp.uniform(
            _name('l1_ratio'),
            0, 1 ) if l1_ratio is None else l1_ratio,
        fit_intercept=hp.pchoice(
            _name('fit_intercept'),
            [ (0.8, True), (0.2, False) ]) if fit_intercept is None else fit_intercept,
        learning_rate='invscaling' if learning_rate is None else learning_rate,
        eta0=hp.loguniform(
            _name('eta0'),
            np.log(1e-5),
            np.log(1e-1)) if eta0 is None else eta0,
        power_t=hp.uniform(
            _name('power_t'),
            0, 1) if power_t is None else power_t,
        n_jobs=n_jobs,
        verbose=verbose,
        )
    return rval

def multinomial_nb(name,
    alpha=None,
    fit_prior=None,
    ):

    def _name(msg):
      return '%s.%s_%s' % (name, 'multinomial_nb', msg)
    

    rval = scope.sklearn_MultinomialNB(
        alpha=hp.quniform(
            _name('alpha'),
            0, 1, 0.001 ) if alpha is None else alpha,
        fit_prior=hp.choice(
            _name('fit_prior'),
            [ True, False ] ) if fit_prior is None else fit_prior,
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
        svc(name + '.svc'),
        sgd(name + '.sgd'),
        knn(name + '.knn', sparse_data=True),
        multinomial_nb(name + '.multinomial_nb')
        ])

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
        whiten=hp_bool(
            name + '.whiten',
            ) if whiten is None else whiten,
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
    
    max_ngram=scope.int( hp.quniform(
        _name('max_ngram'),
        1, 4, 1 ) )

    rval = scope.sklearn_Tfidf(
        stop_words=hp.choice(
            _name('stop_words'),
            [ 'english', None ] ) if analyzer is None else analyzer,
        lowercase=hp_bool(
            _name('lowercase'),
            ) if lowercase is None else lowercase,
        max_df=max_df,
        min_df=min_df,
        binary=hp_bool(
            _name('binary'),
            ) if binary is None else binary,
        ngram_range=(1,max_ngram) if ngram_range is None else ngram_range,
        norm=norm,
        use_idf=use_idf,
        smooth_idf=smooth_idf,
        sublinear_tf=sublinear_tf,
        )
    return rval

def min_max_scaler(name, feature_range=None, copy=True):
    if feature_range is None:
        feature_range = (
            hp.choice(name + '.feature_min', [-1.0, 0.0]),
            1.0)
    rval = scope.sklearn_MinMaxScaler(
        feature_range=feature_range,
        copy=copy,
        )
    return rval


def normalizer(name, norm=None):
    rval = scope.sklearn_Normalizer(
        norm=hp.choice(
            name + '.with_mean',
            ['l1', 'l2'],
            ) if norm is None else norm,
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
        random_state=_random_state(name + '.rstate', random_state),
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
        [],
        # -- not putting in one-hot because it can make vectors huge
        #[one_hot_encoder(name + '.one_hot_encoder')],
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

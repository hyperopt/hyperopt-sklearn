import numpy as np
import sklearn.svm
import sklearn.decomposition
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
def sklearn_PCA(*args, **kwargs):
    return sklearn.decomposition.PCA(*args, **kwargs)


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
    """
    loss_penalty = hp.choice( 'loss_penalty', [ ('l1', 'l2'), 
                                                ('l2', 'l1'),
                                                ('l2', 'l2') ] )

    #loss_penalty[0]
    #loss_penalty[1]
    rval = scope.sklearn_LinearSVC(
        C=hp.lognormal(
            _name('C'),
            np.log(1.0),
            np.log(4.0)) if C is None else C,
        loss=hp.choice(
            _name('loss'),
            ('l1', 'l2')) if loss is None else loss,
        penalty=hp.choice(
            _name('penalty'),
            ('l1', 'l2')) if penalty is None else penalty,
        dual=hp.choice(
            _name('dual'),
            (True, False)) if dual is None else dual,
        tol=hp.lognormal(
            _name('tol'),
            np.log(1e-3),
            np.log(10)) if tol is None else tol,
        multi_class=hp.choice(
            _name('multi_class'),
            ('ovr', 'crammer_singer')) if multi_class is None else multi_class,
        fit_intercept=hp.choice(
            _name('fit_intercept'),
            (True, False)) if fit_intercept is None else fit_intercept,
        verbose=verbose,
        )
    return rval


def any_classifier(name):
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
                np.log(1000),
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


def any_preprocessing(name):
    return hp.choice('%s' % name, [
        [pca(name + '.pca')],
        #rbm(name + '.rbm'),
        ])


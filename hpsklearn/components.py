import numpy as np
import sklearn
from hyperopt.pyll import scope
from hyperopt import hp

@scope.define
def sklearn_SVC(*args, **kwargs):
    return sklearn.svm.SVC(*args, **kwargs)


@scope.define
def sklearn_PCA(*args, **kwargs):
    return sklearn.decompositions.PCA(*args, **kwargs)


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
    cache_size=None,
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
        max_iter=scope.int(
            hp.qlognormal(
                _name('max_iter'),
                np.log(1000),
                np.log(10),
                )) if max_iter is None else max_iter,
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
    cache_size=None,
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
        max_iter=scope.int(
            hp.qlognormal(
                _name('max_iter'),
                np.log(1000),
                np.log(10),
                )) if max_iter is None else max_iter,
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
    cache_size=None,
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
        max_iter=scope.int(
            hp.qlognormal(
                _name('max_iter'),
                np.log(1000),
                np.log(10),
                )) if max_iter is None else max_iter,
        verbose=verbose,
        cache_size=cache_size,
        )


def svc_sigm(name,
    C=None,
    gamma=None,
    coef0=None,
    shrinking=None,
    tol=None,
    max_iter=None,
    verbose=False,
    cache_size=None,
    ):
    """
    Return a pyll graph with hyperparamters that will construct
    a sklearn.svm.SVC model with an RBF kernel.

    """
    def _name(msg):
        return '%s.%s_%s' % (name, 'sigm', msg)

    rval = scope.sklearn_SVC(
        kernel='sigm',
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
        max_iter=scope.int(
            hp.qlognormal(
                _name('max_iter'),
                np.log(1000),
                np.log(10),
                )) if max_iter is None else max_iter,
        verbose=verbose,
        cache_size=cache_size,
        )


def svc(name,
    C=None,
    kernels=['linear', 'rbf', 'poly', 'sigm'],
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
        'sigm': svc_sigm(name,
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


def any_classifier(name):
    return hp.choice('%s', [
        svc(name + '.svc'),
        ])


def pca(name,
    n_components=None,
    whiten=None,
    ):
    rval = scope.sklearn_PCA(
        n_components=scope.int(
            hp.qloguniform(
                name + '.n_components',
                low=0.5,
                high=999.5,
                q=1.0)) if n_components is None else n_components,
        whiten=hp_bool(
            name + '.whiten',
            ) if whiten is None else whiten,
        )
    return rval


def rbm(name):
    return None

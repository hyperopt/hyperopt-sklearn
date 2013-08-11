"""
Sample code used in a couple of talks to illustrate what hyperopt can do in the context of familiar machine learning components.

"""


# SKETCH of how to optimize across sklearn classifiers
import numpy as np
from hyperopt.pyll import scope
from hyperopt import hp
import sklearn.decomposition
import sklearn.mixture
import sklearn.tree
import sklearn.svm
try:
  scope.define(sklearn.decomposition.PCA)
  scope.define(sklearn.mixture.GMM)
  scope.define(sklearn.tree.DecisionTreeClassifier)
  scope.define(sklearn.svm.SVC)
except ValueError:
    # -- already defined these symbols
    pass


# -- (1) DEFINE A SEARCH SPACE
classifier = hp.choice('classifier', [
    scope.DecisionTreeClassifier(
        criterion=hp.choice(
            'dtree_criterion', ['gini', 'entropy']),
        max_features=hp.uniform(
            'dtree_max_features', 0, 1),
        max_depth=hp.quniform(
            'dtree_max_depth', 0, 25, 1)),
    scope.SVC(
        C=hp.lognormal(
            'svc_rbf_C', 0, 3),
        kernel='rbf',
        gamma=hp.lognormal(
            'svc_rbf_gamma', 0, 2),
        tol=hp.lognormal(
            'svc_rbf_tol', np.log(1e-3), 1)),
    ])

pre_processing = hp.choice('preproc_algo', [
    scope.PCA(
        n_components=1 + hp.qlognormal(
            'pca_n_comp', np.log(10), np.log(10), 1),
        whiten=hp.choice(
            'pca_whiten', [False, True])),
    scope.GMM(
        n_components=1 + hp.qlognormal(
            'gmm_n_comp', np.log(100), np.log(10), 1),
        covariance_type=hp.choice(
            'gmm_covtype', ['spherical', 'tied', 'diag', 'full'])),
    ])

sklearn_space = {'pre_processing': pre_processing,
                 'classifier': classifier}
from hyperopt.pyll.stochastic import sample
print sample(sklearn_space)
print sample(sklearn_space)


# -- (2) DEFINE AN OBJECTIVE FUNCTION
def objective(args):
    preprocessing = args['pre_processing']
    classifier = args['classifier']
    X, y = load_data()
    Xpp = preprocessing.transform(X)
    classifier.fit(Xpp, y)
    return {
            'loss': classifier.score(Xpp, y),
            'status': 'ok',
            'foo': 123,           # -- can save more diagnotics
            'other-stuff': None,
            }

# -- (3) Optional: create a database to store experiment results
if 0:
    import hyperopt.mongoexp
    trials = hyperopt.mongoexp.MongoTrials('<server info>')
else:
    trials = None

# -- (4) Find the best model!
# ## THIS FAILS BECAUSE load_data has not been defined, it's just
#    some example code.
hyperopt.fmin(objective,
              sklearn_space,
              algo=hyperopt.rand.suggest,
              trials=trials,
              max_evals=100)


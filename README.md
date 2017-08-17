# hyperopt-sklearn

[Hyperopt-sklearn](http://hyperopt.github.com/hyperopt-sklearn/) is
[Hyperopt](http://hyperopt.github.com/hyperopt)-based model selection among machine learning algorithms in
[scikit-learn](http://scikit-learn.org/).

See how to use hyperopt-sklearn through [examples](http://hyperopt.github.io/hyperopt-sklearn/#documentation)
or older
[notebooks](http://nbviewer.ipython.org/github/hyperopt/hyperopt-sklearn/tree/master/notebooks)


## Installation

Installation from a git clone using pip is supported:

    git clone git@github.com:hyperopt/hyperopt-sklearn.git
    (cd hyperopt-sklearn && pip install -e .)

## Usage

If you are familiar with sklearn, adding the hyperparameter search with hyperopt-sklearn is only a one line change from the standard pipeline.

```
from hpsklearn import HyperoptEstimator, svc
from sklearn import svm

# Load Data
# ...

if use_hpsklearn:
    estim = HyperoptEstimator(classifier=svc('mySVC'))
else:
    estim = svm.SVC()

estim.fit(X_train, y_train)

print(estim.score(X_test, y_test))
# <<show score here>>
```

Complete example using the Iris dataset:

```
from hpsklearn import HyperoptEstimator, any_classifier
from sklearn.datasets import load_iris
from hyperopt import tpe
import numpy as np

# Download the data and split into training and test sets

iris = load_iris()

X = iris.data
y = iris.target

test_size = int(0.2 * len(y))
np.random.seed(13)
indices = np.random.permutation(len(X))
X_train = X[ indices[:-test_size]]
y_train = y[ indices[:-test_size]]
X_test = X[ indices[-test_size:]]
y_test = y[ indices[-test_size:]]

# Instantiate a HyperoptEstimator with the search space and number of evaluations

estim = HyperoptEstimator(classifier=any_classifier('my_clf'),
                          preprocessing=any_preprocessing('my_pre'),
                          algo=tpe.suggest,
                          max_evals=100,
                          trial_timeout=120)

# Search the hyperparameter space based on the data

estim.fit( X_train, y_train )

# Show the results

print( estim.score( X_test, y_test ) )
# 1.0

print( estim.best_model() )
# {'learner': ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
#           max_depth=3, max_features='log2', max_leaf_nodes=None,
#           min_impurity_decrease=0.0, min_impurity_split=None,
#           min_samples_leaf=1, min_samples_split=2,
#           min_weight_fraction_leaf=0.0, n_estimators=13, n_jobs=1,
#           oob_score=False, random_state=1, verbose=False,
#           warm_start=False), 'preprocs': (), 'ex_preprocs': ()}
```

Here's an example using MNIST and being more specific on the classifier and preprocessing.

```
from hpsklearn import HyperoptEstimator, extra_trees
from sklearn.datasets import fetch_mldata
from hyperopt import tpe
import numpy as np

# Download the data and split into training and test sets

digits = fetch_mldata('MNIST original')

X = digits.data
y = digits.target

test_size = int(0.2 * len(y))
np.random.seed(13)
indices = np.random.permutation(len(X))
X_train = X[ indices[:-test_size]]
y_train = y[ indices[:-test_size]]
X_test = X[ indices[-test_size:]]
y_test = y[ indices[-test_size:]]

# Instantiate a HyperoptEstimator with the search space and number of evaluations

estim = HyperoptEstimator(classifier=extra_trees('my_clf'),
                          preprocessing=[],
                          algo=tpe.suggest,
                          max_evals=10,
                          trial_timeout=300)

# Search the hyperparameter space based on the data

estim.fit( X_train, y_train )

# Show the results

print( estim.score( X_test, y_test ) )
# 0.962785714286 

print( estim.best_model() )
# {'learner': ExtraTreesClassifier(bootstrap=True, class_weight=None, criterion='entropy',
#           max_depth=None, max_features=0.959202875857,
#           max_leaf_nodes=None, min_impurity_decrease=0.0,
#           min_impurity_split=None, min_samples_leaf=1,
#           min_samples_split=2, min_weight_fraction_leaf=0.0,
#           n_estimators=20, n_jobs=1, oob_score=False, random_state=3,
#           verbose=False, warm_start=False), 'preprocs': (), 'ex_preprocs': ()}
```

## Available Components

Not all of the classifiers/regressors/preprocessing from sklearn have been implemented yet.
A list of those currently available is shown below.
If there is something you would like that is not on the list, feel free to make an issue or a pull request!
The source code for implementing these functions is found [here](https://github.com/hyperopt/hyperopt-sklearn/blob/master/hpsklearn/components.py)

### Classifiers

```
svc
svc_linear
svc_rbf
svc_poly
svc_sigmoid
liblinear_svc

knn

ada_boost
gradient_boosting

random_forest
extra_trees
decision_tree

sgd

xgboost_classification

multinomial_nb
gaussian_nb

passive_aggressive

linear_discriminant_analysis
quadratic_discriminant_analysis

rbm

colkmeans

one_vs_rest
one_vs_one
output_code

```

For a simple generic search space across many classifiers, use `any_classifier`. If your data is in a sparse matrix format, use `any_sparse_classifier`.

### Regressors

```
svr
svr_linear
svr_rbf
svr_poly
svr_sigmoid

knn_regression

ada_boost_regression
gradient_boosting_regression

random_forest_regression
extra_trees_regression

sgd_regression

xgboost_regression
```

For a simple generic search space across many regressors, use `any_regressor`. If your data is in a sparse matrix format, use `any_sparse_regressor`.

### Preprocessing

```
pca

one_hot_encoder

standard_scaler
min_max_scaler
normalizer

ts_lagselector

tfidf

```

For a simple generic search space across many preprocessing algorithms, use `any_preprocessing`.
If you are working with raw text data, use `any_text_preprocessing`.
Currently only TFIDF is used for text, but more may be added in the future.
Note that the `preprocessing` parameter in `HyperoptEstimator` is expecting a list, since various preprocessing steps can be chained together.
The generic search space functions `any_preprocessing` and `any_text_preprocessing` already return a list, but the others do not so they should be wrapped in a list.
If you do not want to do any preprocessing, pass in an empty list `[]`.

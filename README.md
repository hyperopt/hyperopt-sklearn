# hyperopt-sklearn

[Hyperopt-sklearn](http://hyperopt.github.com/hyperopt-sklearn/) is
[Hyperopt](http://hyperopt.github.com/hyperopt)-based model selection among machine learning algorithms in
[scikit-learn](http://scikit-learn.org/).

See how to use hyperopt-sklearn through [examples](http://hyperopt.github.io/hyperopt-sklearn/#documentation)
or older
[notebooks](http://nbviewer.ipython.org/github/hyperopt/hyperopt-sklearn/tree/master/notebooks)

More examples can be found in the Example Usage section of the SciPy paper

Komer B., Bergstra J., and Eliasmith C. "Hyperopt-Sklearn: automatic hyperparameter configuration for Scikit-learn" Proc. SciPy 2014. http://conference.scipy.org/proceedings/scipy2014/pdfs/komer.pdf

## Installation

Installation from the GitHub repository is supported using pip:

    pip install git+https://github.com/hyperopt/hyperopt-sklearn
    
Optionally you can install a specific tag, branch or commit:

    pip install git+https://github.com/hyperopt/hyperopt-sklearn@0.0.3
    pip install git+https://github.com/hyperopt/hyperopt-sklearn@master
    pip install git+https://github.com/hyperopt/hyperopt-sklearn@fd718c44fc440bd6e2718ec1442b1af58cafcb18

## Usage

If you are familiar with sklearn, adding the hyperparameter search with hyperopt-sklearn is only a one line change from the standard pipeline.

```python
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

Each component comes with a default search space.
The search space for each parameter can be changed or set constant by passing in keyword arguments.
In the following example the `penalty` parameter is held constant during the search, and the `loss` and `alpha` parameters have their search space modified from the default.

```python
from hpsklearn import HyperoptEstimator, sgd
from hyperopt import hp
import numpy as np

sgd_penalty = 'l2'
sgd_loss = hp.pchoice(’loss’, [(0.50, ’hinge’), (0.25, ’log’), (0.25, ’huber’)])
sgd_alpha = hp.loguniform(’alpha’, low=np.log(1e-5), high=np.log(1))

estim = HyperoptEstimator(classifier=sgd(’my_sgd’, penalty=sgd_penalty, loss=sgd_loss, alpha=sgd_alpha))
estim.fit(X_train, y_train)
```

Complete example using the Iris dataset:

```python
from hpsklearn import HyperoptEstimator, any_classifier, any_preprocessing
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
X_train = X[indices[:-test_size]]
y_train = y[indices[:-test_size]]
X_test = X[indices[-test_size:]]
y_test = y[indices[-test_size:]]

# Instantiate a HyperoptEstimator with the search space and number of evaluations

estim = HyperoptEstimator(classifier=any_classifier('my_clf'),
                          preprocessing=any_preprocessing('my_pre'),
                          algo=tpe.suggest,
                          max_evals=100,
                          trial_timeout=120)

# Search the hyperparameter space based on the data

estim.fit(X_train, y_train)

# Show the results

print(estim.score(X_test, y_test))
# 1.0

print(estim.best_model())
# {'learner': ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
#           max_depth=3, max_features='log2', max_leaf_nodes=None,
#           min_impurity_decrease=0.0, min_impurity_split=None,
#           min_samples_leaf=1, min_samples_split=2,
#           min_weight_fraction_leaf=0.0, n_estimators=13, n_jobs=1,
#           oob_score=False, random_state=1, verbose=False,
#           warm_start=False), 'preprocs': (), 'ex_preprocs': ()}
```

Here's an example using MNIST and being more specific on the classifier and preprocessing.

```python
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
X_train = X[indices[:-test_size]]
y_train = y[indices[:-test_size]]
X_test = X[indices[-test_size:]]
y_test = y[indices[-test_size:]]

# Instantiate a HyperoptEstimator with the search space and number of evaluations

estim = HyperoptEstimator(classifier=extra_trees('my_clf'),
                          preprocessing=[],
                          algo=tpe.suggest,
                          max_evals=10,
                          trial_timeout=300)

# Search the hyperparameter space based on the data

estim.fit( X_train, y_train )

# Show the results

print(estim.score(X_test, y_test))
# 0.962785714286 

print(estim.best_model())
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

rbm

colkmeans

```

For a simple generic search space across many preprocessing algorithms, use `any_preprocessing`.
If you are working with raw text data, use `any_text_preprocessing`.
Currently only TFIDF is used for text, but more may be added in the future.
Note that the `preprocessing` parameter in `HyperoptEstimator` is expecting a list, since various preprocessing steps can be chained together.
The generic search space functions `any_preprocessing` and `any_text_preprocessing` already return a list, but the others do not so they should be wrapped in a list.
If you do not want to do any preprocessing, pass in an empty list `[]`.

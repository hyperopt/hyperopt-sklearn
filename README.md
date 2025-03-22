# hyperopt-sklearn

[Hyperopt-sklearn](https://github.com/hyperopt/hyperopt-sklearn) is
[Hyperopt](https://github.com/hyperopt/hyperopt)-based model selection among machine learning algorithms in
[scikit-learn](https://scikit-learn.org/).

See how to use hyperopt-sklearn through [examples](http://hyperopt.github.io/hyperopt-sklearn/#documentation)
More examples can be found in the Example Usage section of the SciPy paper

Komer B., Bergstra J., and Eliasmith C. "Hyperopt-Sklearn: automatic hyperparameter configuration for Scikit-learn" Proc. SciPy 2014. https://proceedings.scipy.org/articles/Majora-14bd3278-006

## Installation

Installation from the GitHub repository is supported using [pip](https://pypi.org/project/hyperopt-sklearn):

    pip install hyperopt-sklearn
    
Optionally you can install a specific tag, branch or commit from the repository:

    pip install git+https://github.com/hyperopt/hyperopt-sklearn@1.0.3
    pip install git+https://github.com/hyperopt/hyperopt-sklearn@master
    pip install git+https://github.com/hyperopt/hyperopt-sklearn@fd718c44fc440bd6e2718ec1442b1af58cafcb18

## Usage

If you are familiar with sklearn, adding the hyperparameter search with hyperopt-sklearn is only a one line change from the standard pipeline.

```python
from hpsklearn import HyperoptEstimator, svc
from sklearn import svm

# Load Data
# ...

if __name__ == "__main__":
    if use_hpsklearn:
        estim = HyperoptEstimator(classifier=svc("mySVC"))
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
from hpsklearn import HyperoptEstimator, sgd_classifier
from hyperopt import hp
import numpy as np

sgd_penalty = "l2"
sgd_loss = hp.pchoice("loss", [(0.50, "hinge"), (0.25, "log"), (0.25, "huber")])
sgd_alpha = hp.loguniform("alpha", low=np.log(1e-5), high=np.log(1))

if __name__ == "__main__":
    estim = HyperoptEstimator(classifier=sgd_classifier("my_sgd", penalty=sgd_penalty, loss=sgd_loss, alpha=sgd_alpha))
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


if __name__ == "__main__":
    # Instantiate a HyperoptEstimator with the search space and number of evaluations
    estim = HyperoptEstimator(classifier=any_classifier("my_clf"),
                              preprocessing=any_preprocessing("my_pre"),
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
from hpsklearn import HyperoptEstimator, extra_tree_classifier
from sklearn.datasets import load_digits
from hyperopt import tpe
import numpy as np

# Download the data and split into training and test sets

digits = load_digits()

X = digits.data
y = digits.target

test_size = int(0.2 * len(y))
np.random.seed(13)
indices = np.random.permutation(len(X))
X_train = X[indices[:-test_size]]
y_train = y[indices[:-test_size]]
X_test = X[indices[-test_size:]]
y_test = y[indices[-test_size:]]


if __name__ == "__main__":
    # Instantiate a HyperoptEstimator with the search space and number of evaluations
    estim = HyperoptEstimator(classifier=extra_tree_classifier("my_clf"),
                              preprocessing=[],
                              algo=tpe.suggest,
                              max_evals=10,
                              trial_timeout=300)

    # Search the hyperparameter space based on the data
    estim.fit(X_train, y_train)

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

Almost all classifiers/regressors/preprocessing scikit-learn components are implemented.
If there is something you would like that is not yet implemented, feel free to make an issue or a pull request!

### Classifiers

```
random_forest_classifier
extra_trees_classifier
bagging_classifier
ada_boost_classifier
gradient_boosting_classifier
hist_gradient_boosting_classifier

bernoulli_nb
categorical_nb
complement_nb
gaussian_nb
multinomial_nb

sgd_classifier
sgd_one_class_svm
ridge_classifier
ridge_classifier_cv
passive_aggressive_classifier
perceptron

dummy_classifier

gaussian_process_classifier

mlp_classifier

linear_svc
nu_svc
svc

decision_tree_classifier
extra_tree_classifier

label_propagation
label_spreading

elliptic_envelope

linear_discriminant_analysis
quadratic_discriminant_analysis

bayesian_gaussian_mixture
gaussian_mixture

k_neighbors_classifier
radius_neighbors_classifier
nearest_centroid

xgboost_classification
lightgbm_classification

one_vs_rest
one_vs_one
output_code
```

For a simple generic search space across many classifiers, use `any_classifier`. 
If your data is in a sparse matrix format, use `any_sparse_classifier`.
For a complete search space across all possible classifiers, use `all_classifiers`.

### Regressors

```
random_forest_regressor
extra_trees_regressor
bagging_regressor
isolation_forest
ada_boost_regressor
gradient_boosting_regressor
hist_gradient_boosting_regressor

linear_regression
bayesian_ridge
ard_regression
lars
lasso_lars
lars_cv
lasso_lars_cv
lasso_lars_ic
lasso
elastic_net
lasso_cv
elastic_net_cv
multi_task_lasso
multi_task_elastic_net
multi_task_lasso_cv
multi_task_elastic_net_cv
poisson_regressor
gamma_regressor
tweedie_regressor
huber_regressor
sgd_regressor
ridge
ridge_cv
logistic_regression
logistic_regression_cv
orthogonal_matching_pursuit
orthogonal_matching_pursuit_cv
passive_aggressive_regressor
quantile_regression
ransac_regression
theil_sen_regressor

dummy_regressor

gaussian_process_regressor

mlp_regressor

cca
pls_canonical
pls_regression

linear_svr
nu_svr
one_class_svm
svr

decision_tree_regressor
extra_tree_regressor

transformed_target_regressor

hp_sklearn_kernel_ridge

bayesian_gaussian_mixture
gaussian_mixture

k_neighbors_regressor
radius_neighbors_regressor

k_means
mini_batch_k_means

xgboost_regression

lightgbm_regression
```

For a simple generic search space across many regressors, use `any_regressor`. 
If your data is in a sparse matrix format, use `any_sparse_regressor`. 
For a complete search space across all possible regressors, use `all_regressors`.

### Preprocessing

```
binarizer
min_max_scaler
max_abs_scaler
normalizer
robust_scaler
standard_scaler
quantile_transformer
power_transformer
one_hot_encoder
ordinal_encoder
polynomial_features
spline_transformer
k_bins_discretizer

tfidf_vectorizer
hashing_vectorizer
count_vectorizer

pca

ts_lagselector

colkmeans
```

For a simple generic search space across many preprocessing algorithms, use `any_preprocessing`.
If your data is in a sparse matrix format, use `any_sparse_preprocessing`.
For a complete search space across all preprocessing algorithms, use `all_preprocessing`.
If you are working with raw text data, use `any_text_preprocessing`.
Currently, only TFIDF is used for text, but more may be added in the future.

Note that the `preprocessing` parameter in `HyperoptEstimator` is expecting a list, since various preprocessing steps can be chained together.
The generic search space functions `any_preprocessing` and `any_text_preprocessing` already return a list, but the others do not, so they should be wrapped in a list.
If you do not want to do any preprocessing, pass in an empty list `[]`.

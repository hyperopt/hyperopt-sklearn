from ._base import \
    linear_regression

from ._bayes import \
    bayesian_ridge, \
    ard_regression

from ._least_angle import \
    lars, \
    lasso_lars, \
    lars_cv, \
    lasso_lars_cv, \
    lasso_lars_ic

from ._coordinate_descent import \
    lasso, \
    elastic_net, \
    lasso_cv, \
    elastic_net_cv, \
    multi_task_lasso, \
    multi_task_elastic_net, \
    multi_task_lasso_cv, \
    multi_task_elastic_net_cv

from ._glm import \
    poisson_regressor, \
    gamma_regressor, \
    tweedie_regressor

from ._huber import \
    huber_regressor

from ._stochastic_gradient import \
    sgd_classifier, \
    sgd_regressor, \
    sgd_one_class_svm

from ._ridge import \
    ridge, \
    ridge_cv, \
    ridge_classifier, \
    ridge_classifier_cv

from ._logistic import \
    logistic_regression, \
    logistic_regression_cv

from ._omp import \
    orthogonal_matching_pursuit, \
    orthogonal_matching_pursuit_cv

from ._passive_aggressive import \
    passive_aggressive_classifier, \
    passive_aggressive_regressor

from ._perceptron import \
    perceptron

from ._quantile import \
    quantile_regression

from ._ransac import \
    ransac_regression

from ._theil_sen import \
    theil_sen_regressor

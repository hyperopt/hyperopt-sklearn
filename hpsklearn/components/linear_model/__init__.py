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

from .ensemble import \
    random_forest_classifier, \
    random_forest_regressor, \
    extra_trees_classifier, \
    extra_trees_regressor, \
    bagging_classifier, \
    bagging_regressor, \
    isolation_forest, \
    ada_boost_classifier, \
    ada_boost_regressor, \
    gradient_boosting_classifier, \
    gradient_boosting_regression, \
    hist_gradient_boosting_classifier, \
    hist_gradient_boosting_regressor

from .preprocessing import \
    binarizer, \
    min_max_scaler, \
    max_abs_scaler, \
    normalizer, \
    robust_scaler, \
    standard_scaler, \
    quantile_transformer, \
    power_transformer, \
    one_hot_encoder, \
    ordinal_encoder, \
    polynomial_features, \
    spline_transformer, \
    k_bins_discretizer

from .naive_bayes import \
    bernoulli_nb, \
    categorical_nb, \
    complement_nb, \
    gaussian_nb, \
    multinomial_nb

from .linear_model import \
    linear_regression, \
    bayesian_ridge, \
    ard_regression, \
    lars, \
    lasso_lars, \
    lars_cv, \
    lasso_lars_cv, \
    lasso_lars_ic, \
    lasso, \
    elastic_net, \
    lasso_cv, \
    elastic_net_cv, \
    multi_task_lasso, \
    multi_task_elastic_net, \
    multi_task_lasso_cv, \
    multi_task_elastic_net_cv, \
    poisson_regressor, \
    gamma_regressor, \
    tweedie_regressor, \
    huber_regressor, \
    sgd_classifier, \
    sgd_regressor, \
    sgd_one_class_svm, \
    ridge, \
    ridge_cv, \
    ridge_classifier, \
    ridge_classifier_cv, \
    logistic_regression, \
    logistic_regression_cv, \
    orthogonal_matching_pursuit, \
    orthogonal_matching_pursuit_cv, \
    passive_aggressive_classifier, \
    passive_aggressive_regressor, \
    perceptron, \
    quantile_regression, \
    ransac_regression, \
    theil_sen_regressor

from .dummy import \
    dummy_classifier, \
    dummy_regressor

from .gaussian_process import \
    gaussian_process_classifier, \
    gaussian_process_regressor

from .neural_network import \
    mlp_classifier, \
    mlp_regressor

from .cross_decomposition import \
    cca, \
    pls_canonical, \
    pls_regression

from .svm import \
    linear_svc, \
    linear_svr, \
    nu_svc, \
    nu_svr, \
    one_class_svm, \
    svc, \
    svr

from .tree import \
    decision_tree_classifier, \
    decision_tree_regressor, \
    extra_tree_classifier, \
    extra_tree_regressor

from .semi_supervised import \
    label_propagation, \
    label_spreading

from .compose import transformed_target_regressor

from .covariance import elliptic_envelope

from .discriminant_analysis import \
    linear_discriminant_analysis, \
    quadratic_discriminant_analysis

from .kernel_ridge import hp_sklearn_kernel_ridge

from .mixture import \
    bayesian_gaussian_mixture, \
    gaussian_mixture

from .neighbors import \
    k_neighbors_classifier, \
    radius_neighbors_classifier, \
    nearest_centroid, \
    k_neighbors_regressor, \
    radius_neighbors_regressor

from .cluster import \
    k_means, \
    mini_batch_k_means


# Legacy any classifier
def any_classifier(name):
    from hyperopt import hp

    classifiers = [
        svc(name + ".svc"),
        # knn(name + ".knn"),
        random_forest_classifier(name + ".random_forest"),
        extra_tree_classifier(name + ".extra_trees"),
        ada_boost_classifier(name + ".ada_boost"),
        gradient_boosting_classifier(name + ".grad_boosting", loss="deviance"),
        sgd_classifier(name + ".sgd")
    ]

    return hp.choice(name, classifiers)

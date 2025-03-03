import warnings
from hyperopt import hp

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
    gradient_boosting_regressor, \
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

from .xgboost import \
    xgboost_classification, \
    xgboost_regression

from .lightgbm import \
    lightgbm_classification, \
    lightgbm_regression

from .feature_extraction import \
    tfidf_vectorizer, \
    hashing_vectorizer, \
    count_vectorizer

from .decomposition import pca

from .lagselectors import ts_lagselector

from .vkmeans import colkmeans

from .ensemble import \
    forest_classifiers, \
    forest_regressors


def xg_boost_check():
    try:
        import xgboost
        return True
    except ImportError:
        warnings.warn("xgboost not installed. Skipping xgboost_classification and xgboost_regression.")
        return False


def lightgbm_check():
    try:
        import lightgbm
        return True
    except ImportError:
        warnings.warn("lightgbm not installed. Skipping lightgbm_classification and lightgbm_regression.")
        return False


# Legacy any classifier
def any_classifier(name):
    """
    Any classifier
    """
    classifiers = [
        svc(name + ".svc"),
        k_neighbors_classifier(name + ".knn"),
        random_forest_classifier(name + ".random_forest"),
        extra_tree_classifier(name + ".extra_trees"),
        ada_boost_classifier(name + ".ada_boost"),
        gradient_boosting_classifier(name + ".grad_boosting"),
        sgd_classifier(name + ".sgd")
    ]

    if xgboost.xgboost:  # noqa
        classifiers.append(xgboost_classification(name + ".xgboost"))

    return hp.choice(name, classifiers)


# Legacy any sparse classifier
def any_sparse_classifier(name):
    """
    Any sparse classifier.
    """
    sparse_classifiers = [
        linear_svc(name + ".linear_svc"),
        sgd_classifier(name + ".sgd"),
        k_neighbors_classifier(name + ".knn", p=2),
        multinomial_nb(name + ".multinomial_nb")
    ]

    return hp.choice(name, sparse_classifiers)


# Legacy any regressor
def any_regressor(name):
    """
    Any regressor
    """
    regressors = [
        svr(name + ".svr"),
        k_neighbors_regressor(name + ".knn"),
        random_forest_regressor(name + ".random_forest"),
        extra_tree_regressor(name + ".extra_trees"),
        ada_boost_regressor(name + ".ada_boost"),
        gradient_boosting_regressor(name + ".grad_boosting"),
        sgd_regressor(name + ".sgd")
    ]

    if xgboost.xgboost:  # noqa
        regressors.append(xgboost_regression(name + ".xgboost"))

    return hp.choice(name, regressors)


# Legacy any sparse regressor
def any_sparse_regressor(name):
    """
    Any sparse regressor.
    """
    sparse_regressors = [
        sgd_regressor(name + ".sgd"),
        k_neighbors_regressor(name + ".knn", p=2)
    ]

    return hp.choice(name, sparse_regressors)


# Legacy any text pre-processing
def any_text_preprocessing(name):
    """
    Generic pre-processing appropriate for text data
    """
    return hp.choice(name, [
        [tfidf_vectorizer(name + ".tfidf")],
        [hashing_vectorizer(name + ".hashing")],
        [count_vectorizer(name + ".count")],
    ])


# Legacy any pre-processing as proposed in #137
def any_preprocessing(name):
    """
    Generic pre-processing appropriate for a wide variety of data
    """
    return hp.choice(name, [
        [pca(name + ".pca")],
        [standard_scaler(name + ".standard_scaler")],
        [min_max_scaler(name + ".min_max_scaler")],
        [normalizer(name + ".normalizer")],
    ])


# Legacy any sparse pre-processing
def any_sparse_preprocessing(name):
    """
    Generic pre-processing appropriate for sparse data
    """
    return hp.choice(name, [
        [standard_scaler(name + ".standard_scaler", with_mean=False)],
        [normalizer(name + ".normalizer")],
    ])


def all_classifiers(name):
    """
    All classifiers
    """
    classifiers = [
        random_forest_classifier(name + ".random_forest"),
        extra_trees_classifier(name + ".extra_trees"),
        bagging_classifier(name + ".bagging"),
        ada_boost_classifier(name + ".ada_boost"),
        gradient_boosting_classifier(name + ".grad_boosting"),
        hist_gradient_boosting_classifier(name + ".hist_grad_boosting"),
        bernoulli_nb(name + ".bernoulli_nb"),
        categorical_nb(name + ".categorical_nb"),
        complement_nb(name + ".complement_nb"),
        gaussian_nb(name + ".gaussian_nb"),
        multinomial_nb(name + ".multinomial_nb"),
        sgd_classifier(name + ".sgd"),
        sgd_one_class_svm(name + ".sgd_one_class_svm"),
        ridge_classifier(name + ".ridge"),
        ridge_classifier_cv(name + ".ridge_cv"),
        logistic_regression(name + ".logistic_regression"),
        logistic_regression_cv(name + ".logistic_regression_cv"),
        passive_aggressive_classifier(name + ".passive_aggressive"),
        perceptron(name + ".perceptron"),
        dummy_classifier(name + ".dummy"),
        gaussian_process_classifier(name + ".gaussian_process"),
        mlp_classifier(name + ".mlp"),
        linear_svc(name + ".linear_svc"),
        nu_svc(name + ".nu_svc"),
        svc(name + ".svc"),
        decision_tree_classifier(name + ".decision_tree"),
        extra_tree_classifier(name + ".extra_tree"),
        label_propagation(name + ".label_propagation"),
        label_spreading(name + ".label_spreading"),
        elliptic_envelope(name + ".elliptic_envelope"),
        linear_discriminant_analysis(name + ".linear_discriminant_analysis"),
        quadratic_discriminant_analysis(name + ".quadratic_discriminant_analysis"),
        bayesian_gaussian_mixture(name + ".bayesian_gaussian_mixture"),
        gaussian_mixture(name + ".gaussian_mixture"),
        k_neighbors_classifier(name + ".knn"),
        radius_neighbors_classifier(name + ".radius_neighbors"),
        nearest_centroid(name + ".nearest_centroid"),
    ]

    if xg_boost_check():
        classifiers.append(xgboost_classification(name + ".xgboost"))

    if lightgbm_check():
        classifiers.append(lightgbm_classification(name + ".lightgbm"))

    return hp.choice(name, classifiers)


def all_regressors(name):
    """
    All regressors
    """
    regressors = [
        random_forest_regressor(name + ".random_forest"),
        extra_trees_regressor(name + ".extra_trees"),
        bagging_regressor(name + ".bagging"),
        isolation_forest(name + ".isolation_forest"),
        ada_boost_regressor(name + ".ada_boost"),
        gradient_boosting_regressor(name + ".grad_boosting"),
        hist_gradient_boosting_regressor(name + ".hist_grad_boosting"),
        linear_regression(name + ".linear_regression"),
        bayesian_ridge(name + ".bayesian_ridge"),
        ard_regression(name + ".ard"),
        lars(name + ".lars"),
        lasso_lars(name + ".lasso_lars"),
        lars_cv(name + ".lars_cv"),
        lasso_lars_cv(name + ".lasso_lars_cv"),
        lasso_lars_ic(name + ".lasso_lars_ic"),
        lasso(name + ".lasso"),
        elastic_net(name + ".elastic_net"),
        lasso_cv(name + ".lasso_cv"),
        elastic_net_cv(name + ".elastic_net_cv"),
        multi_task_lasso(name + ".multi_task_lasso"),
        multi_task_elastic_net(name + ".multi_task_elastic_net"),
        multi_task_lasso_cv(name + ".multi_task_lasso_cv"),
        multi_task_elastic_net_cv(name + ".multi_task_elastic_net_cv"),
        poisson_regressor(name + ".poisson_regressor"),
        gamma_regressor(name + ".gamma_regressor"),
        tweedie_regressor(name + ".tweedie_regressor"),
        huber_regressor(name + ".huber_regressor"),
        sgd_regressor(name + ".sgd"),
        ridge(name + ".ridge"),
        ridge_cv(name + ".ridge_cv"),
        orthogonal_matching_pursuit(name + ".orthogonal_matching_pursuit"),
        orthogonal_matching_pursuit_cv(name + ".orthogonal_matching_pursuit_cv"),
        passive_aggressive_regressor(name + ".passive_aggressive"),
        quantile_regression(name + ".quantile_regression"),
        ransac_regression(name + ".ransac_regression"),
        theil_sen_regressor(name + ".theil_sen_regressor"),
        dummy_regressor(name + ".dummy"),
        gaussian_process_regressor(name + ".gaussian_process"),
        mlp_regressor(name + ".mlp"),
        cca(name + ".cca"),
        pls_canonical(name + ".pls_canonical"),
        pls_regression(name + ".pls_regression"),
        linear_svr(name + ".linear_svr"),
        nu_svr(name + ".nu_svr"),
        one_class_svm(name + ".one_class_svm"),
        svr(name + ".svr"),
        decision_tree_regressor(name + ".decision_tree"),
        extra_tree_regressor(name + ".extra_tree"),
        transformed_target_regressor(name + ".transformed_target"),
        hp_sklearn_kernel_ridge(name + ".hp_sklearn_kernel_ridge"),
        bayesian_gaussian_mixture(name + ".bayesian_gaussian_mixture"),
        gaussian_mixture(name + ".gaussian_mixture"),
        k_neighbors_regressor(name + ".knn"),
        radius_neighbors_regressor(name + ".radius_neighbors"),
        k_means(name + ".k_means"),
        mini_batch_k_means(name + ".mini_batch_k_means"),
    ]

    if xg_boost_check():
        regressors.append(xgboost_regression(name + ".xgboost"))

    if lightgbm_check():
        regressors.append(lightgbm_regression(name + ".lightgbm"))

    return hp.choice(name, regressors)


def all_preprocessing(name):
    """
    All pre-processing
    """
    preprocessors = [
        [binarizer(name + ".binarizer")],
        [min_max_scaler(name + ".min_max_scaler")],
        [max_abs_scaler(name + ".max_abs_scaler")],
        [normalizer(name + ".normalizer")],
        [robust_scaler(name + ".robust_scaler")],
        [standard_scaler(name + ".standard_scaler")],
        [quantile_transformer(name + ".quantile_transformer")],
        [power_transformer(name + ".power_transformer")],
        [one_hot_encoder(name + ".one_hot_encoder")],
        [ordinal_encoder(name + ".ordinal_encoder")],
        [polynomial_features(name + ".polynomial_features")],
        [spline_transformer(name + ".spline_transformer")],
        [k_bins_discretizer(name + ".k_bins_discretizer")],
        [tfidf_vectorizer(name + ".tfidf")],
        [hashing_vectorizer(name + ".hashing")],
        [count_vectorizer(name + ".count")],
        [pca(name + ".pca")],
        [ts_lagselector(name + ".ts_lagselector")],
        [colkmeans(name + ".colkmeans")],
    ]

    return hp.choice(name, preprocessors)

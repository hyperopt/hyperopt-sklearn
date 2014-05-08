
def test_demo_iris():
    import numpy as np
    import skdata.iris.view
    import hyperopt.tpe
    import hpsklearn

    data_view = skdata.iris.view.KfoldClassification(4)

    estimator = hpsklearn.estimator(
        preprocessing=hpsklearn.components.any_preprocessing('pp'),
        classifier=hpsklearn.components.any_classifier('clf'),
        algo=hyperopt.tpe,
        trial_timeout=15.0, # seconds
        max_evals=100,
        )

    # /BEGIN `Demo version of estimator.fit()`

    iterator = estimator.fit_iter(
        data_view.split[0].train.X,
        data_view.split[0].train.y)
    iterator.next()

    while len(estimator.trials.trials) < estimator.max_evals:
        iterator.send(1) # -- try one more model
        hpsklearn.demo_support.scatter_error_vs_time(estimator)
        hpsklearn.demo_support.bar_classifier_choice(estimator)

    estimator.retrain_best_model_on_full_data(
        data_view.split[0].train.X,
        data_view.split[0].train.y)

    # /END Demo version of `estimator.fit()`

    test_predictions = estimator.predict(data_view.split[0].test.X)
    print np.mean(test_predictions == data_view.split[0].test.y)

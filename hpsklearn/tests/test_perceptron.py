import sklearn.linear_model
import skdata.iris.view
from skdata.base import SklearnClassifier
from hpsklearn.perceptron import AutoPerceptron


def test_basic():
    view = skdata.iris.view.KfoldClassification(4)

    # fit a model using the default Perceptron args:
    def fit_default():
        algo = SklearnClassifier(sklearn.linear_model.Perceptron)
        mean_test_error = view.protocol(algo)

        assert len(algo.results['loss']) == 4
        assert len(algo.results['best_model']) == 4

        for loss_report in algo.results['loss']:
            print loss_report['task_name'] + \
                (": err = %0.3f" % (loss_report['err_rate']))
        return mean_test_error

    default_test_error = fit_default()

    algo = SklearnClassifier(AutoPerceptron)
    mean_test_error = view.protocol(algo)

    assert len(algo.results['loss']) == 4
    assert len(algo.results['best_model']) == 4

    for loss_report in algo.results['loss']:
        print loss_report['task_name'] + \
            (": err = %0.3f" % (loss_report['err_rate']))

    print "MEAN OPT ERROR", mean_test_error
    print "MEAN DEFAULT ERROR", default_test_error
    assert mean_test_error <= default_test_error

    assert mean_test_error < 0.15 # 0.147 on Feb 19, 2013
    assert default_test_error > 0.3 # 0.305 on Feb 19, 2013


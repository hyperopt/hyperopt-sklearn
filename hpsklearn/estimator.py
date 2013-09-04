
class HyperoptEstimatorFactory(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self):
        raise NotImplementedError()

    def fit(self):
        pass

    def predict(self, X):
        raise NotImplementedError()


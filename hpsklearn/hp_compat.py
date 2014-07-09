from searchspaces import variable


def uniform(label, low, high):
    return variable(label, value_type=float, distribution='uniform',
                    minimum=low, maximum=high)


def loguniform(label, low, high):
    return variable(label, value_type=float, distribution='loguniform',
                    minimum=low, maximum=high)


def quniform(label, low, high, q):
    return variable(label, value_type=float, distribution='quniform',
                    minimum=low, maximum=high, q=q)


def qloguniform(label, low, high, q):
    return variable(label, value_type=float, distribution='qloguniform',
                    minimum=low, maximum=high, q=q)


def normal(label, mu, sigma):
    return variable(label, value_type=float, distribution='normal',
                    mu=mu, sigma=sigma)


def qnormal(label, mu, sigma, q):
    return variable(label, value_type=float, distribution='qnormal',
                    mu=mu, sigma=sigma, q=q)


def lognormal(label, mu, sigma):
    return variable(label, value_type=float, distribution='lognormal',
                    mu=mu, sigma=sigma)


def qlognormal(label, mu, sigma, q):
    return variable(label, value_type=float, distribution='qlognormal',
                    mu=mu, sigma=sigma, q=q)


def randint(label, upper):
    return variable(label, value_type=int, distribution='randint',
                    maximum=upper - 1)


def categorical(label, p, upper=None):
    return variable(label, value_type=range(len(p)),
                    distribution='categorical', p=p)

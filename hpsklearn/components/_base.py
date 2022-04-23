import functools


def validate(params, validation_test, msg):
    """
    Validation decorator for parameter checks
     includes automatic value error raising
     allows for multiple usages per function
    """
    def inner(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """
            Iterate over keyword arguments
             if keyword in parameter perform lambda test
            """
            for k, v in kwargs.items():
                if k in params and not validation_test(v):
                    raise ValueError(msg % (k, v))

            return func(*args, **kwargs)
        return wrapper
    return inner

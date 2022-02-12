import hyperopt
import time
import typing


class _NonFiniteFeature(Exception):
    """
    Called after finite check
     used for exception handling
    """


def _custom_handler(str_exc: str, t_start: float, exc) -> typing.Tuple[Exception, str]:
    """
    Custom exception handler to reduce duplicated code.
    """
    if str_exc in str(exc):
        rval = {
            "status": hyperopt.STATUS_FAIL,
            "failure": str(exc),
            "duration": time.time() - t_start
        }
        rtype = "return"
    else:
        rval = exc
        rtype = "raise"

    return rval, rtype

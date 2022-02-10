import unittest

from hpsklearn import quantile_regression
from tests.utils import \
    StandardRegressorTest, \
    generate_attributes


class QuantileRegressorTest(StandardRegressorTest):
    """
    Class for _quantile regression testing
    """


generate_attributes(
    TestClass=QuantileRegressorTest,
    fn_list=[quantile_regression],
    is_classif=False,
    trial_timeout=30.0,  # Increase timeout for longer runtime due to some solvers
)


if __name__ == '__main__':
    unittest.main()

import unittest

from hpsklearn import \
    huber_regressor
from hpsklearn.tests.utils import \
    StandardRegressorTest, \
    generate_test_attributes


class TestHuberRegression(StandardRegressorTest):
    """
    Class for _huber regression testing
    """


generate_test_attributes(
    TestClass=TestHuberRegression,
    fn_list=[huber_regressor],
    is_classif=False
)


if __name__ == '__main__':
    unittest.main()

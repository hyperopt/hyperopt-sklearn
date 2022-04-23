import unittest

from hpsklearn import \
    huber_regressor
from tests.utils import \
    StandardRegressorTest, \
    generate_attributes


class TestHuberRegression(StandardRegressorTest):
    """
    Class for _huber regression testing
    """


generate_attributes(
    TestClass=TestHuberRegression,
    fn_list=[huber_regressor],
    is_classif=False
)


if __name__ == '__main__':
    unittest.main()

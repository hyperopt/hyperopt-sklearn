import unittest

from hpsklearn import ransac_regression
from tests.utils import \
    StandardRegressorTest, \
    generate_attributes


class TestRANSACRegressor(StandardRegressorTest):
    """
    Class for _ransac regression testing
    """


generate_attributes(
    TestClass=TestRANSACRegressor,
    fn_list=[ransac_regression],
    is_classif=False,
)


if __name__ == '__main__':
    unittest.main()

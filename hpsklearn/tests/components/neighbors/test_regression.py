import unittest

from hpsklearn import \
    k_neighbors_regressor, \
    radius_neighbors_regressor
from hpsklearn.tests.utils import \
    StandardRegressorTest, \
    generate_test_attributes


class TestNeighborsRegression(StandardRegressorTest):
    """
    Class for _regression regression testing
    """


generate_test_attributes(
    TestClass=TestNeighborsRegression,
    fn_list=[k_neighbors_regressor, radius_neighbors_regressor],
    is_classif=False,
)


if __name__ == '__main__':
    unittest.main()

import unittest

from hpsklearn import \
    k_neighbors_regressor, \
    radius_neighbors_regressor
from tests.utils import \
    StandardRegressorTest, \
    generate_attributes


class TestNeighborsRegression(StandardRegressorTest):
    """
    Class for _regression regression testing
    """


generate_attributes(
    TestClass=TestNeighborsRegression,
    fn_list=[k_neighbors_regressor, radius_neighbors_regressor],
    is_classif=False,
)


if __name__ == '__main__':
    unittest.main()

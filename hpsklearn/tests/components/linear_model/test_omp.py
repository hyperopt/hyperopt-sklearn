import unittest

from hpsklearn import \
    orthogonal_matching_pursuit, \
    orthogonal_matching_pursuit_cv
from hpsklearn.tests.utils import \
    StandardRegressorTest, \
    generate_test_attributes


class TestOMPRegression(StandardRegressorTest):
    """
    Class for _omp regression testing
    """


generate_test_attributes(
    TestClass=TestOMPRegression,
    fn_list=[orthogonal_matching_pursuit, orthogonal_matching_pursuit_cv],
    is_classif=False,
)


if __name__ == '__main__':
    unittest.main()

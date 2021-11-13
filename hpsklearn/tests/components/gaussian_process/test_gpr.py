import unittest

from hpsklearn import gaussian_process_regressor
from hpsklearn.tests.utils import \
    StandardRegressorTest, \
    generate_test_attributes


class TestGaussianProcessRegressor(StandardRegressorTest):
    """
    Class for _gpr regression testing
    """


generate_test_attributes(
    TestClass=TestGaussianProcessRegressor,
    fn_list=[gaussian_process_regressor],
    is_classif=False
)


if __name__ == '__main__':
    unittest.main()

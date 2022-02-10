import unittest

from hpsklearn import gaussian_process_regressor
from tests.utils import \
    StandardRegressorTest, \
    generate_attributes


class TestGaussianProcessRegressor(StandardRegressorTest):
    """
    Class for _gpr regression testing
    """


generate_attributes(
    TestClass=TestGaussianProcessRegressor,
    fn_list=[gaussian_process_regressor],
    is_classif=False
)


if __name__ == '__main__':
    unittest.main()

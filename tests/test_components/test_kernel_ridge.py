import unittest

from hpsklearn import hp_sklearn_kernel_ridge
from tests.utils import \
    StandardRegressorTest, \
    generate_attributes


class TestKernelRidgeRegression(StandardRegressorTest):
    """
    Class for _kernel_ridge regression testing
    """


generate_attributes(
    TestClass=TestKernelRidgeRegression,
    fn_list=[hp_sklearn_kernel_ridge],
    is_classif=False,
    non_negative_input=True,
)


if __name__ == '__main__':
    unittest.main()

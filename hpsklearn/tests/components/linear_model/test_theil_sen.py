import unittest

from hpsklearn import theil_sen_regressor
from hpsklearn.tests.utils import \
    StandardRegressorTest, \
    generate_test_attributes


class TestTheilSenRegressor(StandardRegressorTest):
    """
    Class for _theil_sen regression testing
    """


generate_test_attributes(
    TestClass=TestTheilSenRegressor,
    fn_list=[theil_sen_regressor],
    is_classif=False,
)


if __name__ == '__main__':
    unittest.main()

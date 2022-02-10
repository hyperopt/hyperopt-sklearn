import unittest

from hpsklearn import \
    lars, \
    lasso_lars, \
    lars_cv, \
    lasso_lars_cv, \
    lasso_lars_ic
from tests.utils import \
    StandardRegressorTest, \
    generate_attributes


class TestLeastAngleRegression(StandardRegressorTest):
    """
    Class for _least_angle regression testing
    """


# List of _least_angle regressors to test
regressors = [
    lars,
    lasso_lars,
    lars_cv,
    lasso_lars_cv,
    lasso_lars_ic
]


generate_attributes(
    TestClass=TestLeastAngleRegression,
    fn_list=regressors,
    is_classif=False
)


if __name__ == '__main__':
    unittest.main()

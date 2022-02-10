import unittest

from hpsklearn import \
    lasso, \
    elastic_net, \
    lasso_cv, \
    elastic_net_cv, \
    multi_task_lasso, \
    multi_task_elastic_net, \
    multi_task_lasso_cv, \
    multi_task_elastic_net_cv
from tests.utils import \
    StandardRegressorTest, \
    MultiTaskRegressorTest, \
    generate_attributes


class TestCoordinateDescentRegression(StandardRegressorTest):
    """
    Class for _coordinate_descent regression testing
    """


class TestCoordinateDescentMultiTaskRegression(MultiTaskRegressorTest):
    """
    Class for _coordinate_descent multi task regression testing
    """


# List of _coordinate_descent regressors to test
regressors = [
    lasso,
    elastic_net,
    lasso_cv,
    elastic_net_cv
]

# List of _coordinate_descent multi task regressors to test
multi_task_regressors = [
    multi_task_lasso,
    multi_task_elastic_net,
    multi_task_lasso_cv,
    multi_task_elastic_net_cv
]


generate_attributes(
    TestClass=TestCoordinateDescentRegression,
    fn_list=regressors,
    is_classif=False
)


generate_attributes(
    TestClass=TestCoordinateDescentMultiTaskRegression,
    fn_list=multi_task_regressors,
    is_classif=False
)


if __name__ == '__main__':
    unittest.main()

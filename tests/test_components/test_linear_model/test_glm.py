import unittest

from hpsklearn import \
    poisson_regressor, \
    gamma_regressor, \
    tweedie_regressor
from tests.utils import \
    StandardRegressorTest, \
    generate_attributes


class TestGLMRegression(StandardRegressorTest):
    """
    Class for _glm regression testing
    """


# List of _glm regressors to test
regressors = [
    poisson_regressor,
    gamma_regressor,
    tweedie_regressor
]


generate_attributes(
    TestClass=TestGLMRegression,
    fn_list=regressors,
    is_classif=False,
    non_negative_output=True
)


if __name__ == '__main__':
    unittest.main()

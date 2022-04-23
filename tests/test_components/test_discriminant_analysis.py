import unittest

from hpsklearn import \
    linear_discriminant_analysis, \
    quadratic_discriminant_analysis
from tests.utils import \
    StandardClassifierTest, \
    generate_attributes


class TestLinearDiscriminantAnalysisClassifier(StandardClassifierTest):
    """
    Class for _discriminant_analysis classification testing
    """


generate_attributes(
    TestClass=TestLinearDiscriminantAnalysisClassifier,
    fn_list=[linear_discriminant_analysis, quadratic_discriminant_analysis],
    is_classif=True,
)


if __name__ == '__main__':
    unittest.main()

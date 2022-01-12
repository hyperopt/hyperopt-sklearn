import unittest

from hpsklearn import gaussian_process_classifier
from tests.utils import \
    StandardClassifierTest, \
    generate_attributes


class TestGaussianProcessClassifier(StandardClassifierTest):
    """
    Class for _gpc classification testing
    """


generate_attributes(
    TestClass=TestGaussianProcessClassifier,
    fn_list=[gaussian_process_classifier],
    is_classif=True
)


if __name__ == '__main__':
    unittest.main()

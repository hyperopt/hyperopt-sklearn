import unittest

from hpsklearn import nearest_centroid
from hpsklearn.tests.utils import \
    StandardClassifierTest, \
    generate_test_attributes


class TestNearestCentroidClassifier(StandardClassifierTest):
    """
    Class for _nearest_centroid classification testing
    """


generate_test_attributes(
    TestClass=TestNearestCentroidClassifier,
    fn_list=[nearest_centroid],
    is_classif=True,
)


if __name__ == '__main__':
    unittest.main()
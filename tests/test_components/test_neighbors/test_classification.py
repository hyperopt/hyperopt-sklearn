import unittest

from hpsklearn import \
    k_neighbors_classifier, \
    radius_neighbors_classifier
from tests.utils import \
    StandardClassifierTest, \
    generate_attributes


class TestNeighborsClassifier(StandardClassifierTest):
    """
    Class for _classification classification testing
    """


generate_attributes(
    TestClass=TestNeighborsClassifier,
    fn_list=[k_neighbors_classifier, radius_neighbors_classifier],
    is_classif=True,
)


if __name__ == '__main__':
    unittest.main()

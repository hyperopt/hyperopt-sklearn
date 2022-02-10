import unittest

from hpsklearn import \
    label_propagation, \
    label_spreading
from tests.utils import \
    StandardClassifierTest, \
    generate_attributes


class TestLabelPropagationClassifier(StandardClassifierTest):
    """
    Class for _label_propagation classification testing
    """


generate_attributes(
    TestClass=TestLabelPropagationClassifier,
    fn_list=[label_propagation, label_spreading],
    is_classif=True,
)


if __name__ == '__main__':
    unittest.main()

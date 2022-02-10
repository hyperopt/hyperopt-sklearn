import unittest

from hpsklearn import perceptron
from tests.utils import \
    StandardClassifierTest, \
    generate_attributes


class TestPerceptronClassifier(StandardClassifierTest):
    """
    Class for _perceptron classification testing
    """


generate_attributes(
    TestClass=TestPerceptronClassifier,
    fn_list=[perceptron],
    is_classif=True,
)


if __name__ == '__main__':
    unittest.main()

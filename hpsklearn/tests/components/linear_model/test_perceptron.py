import unittest

from hpsklearn import perceptron
from hpsklearn.tests.utils import \
    StandardClassifierTest, \
    generate_test_attributes


class TestPerceptronClassifier(StandardClassifierTest):
    """
    Class for _perceptron classification testing
    """


generate_test_attributes(
    TestClass=TestPerceptronClassifier,
    fn_list=[perceptron],
    is_classif=True,
)


if __name__ == '__main__':
    unittest.main()

import unittest

from tests.utils import \
    StandardClassifierTest, \
    generate_attributes
from hpsklearn import \
    bernoulli_nb, \
    categorical_nb, \
    complement_nb, \
    gaussian_nb, \
    multinomial_nb


class TestNaiveBayes(StandardClassifierTest):
    """
    Class for naive_bayes classification testing
    """


generate_attributes(
    TestClass=TestNaiveBayes,
    fn_list=[bernoulli_nb, gaussian_nb],
    is_classif=True
)


generate_attributes(
    TestClass=TestNaiveBayes,
    fn_list=[categorical_nb, complement_nb, multinomial_nb],
    is_classif=True,
    non_negative_input=True
)


if __name__ == '__main__':
    unittest.main()

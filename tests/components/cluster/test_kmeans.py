import unittest

from hpsklearn import \
    k_means, \
    mini_batch_k_means
from tests.utils import \
    generate_attributes, \
    IrisTest


class TestKMeansIris(IrisTest):
    """
    Class for _kmeans iris testing
    """


generate_attributes(
    TestClass=TestKMeansIris,
    fn_list=[k_means, mini_batch_k_means],
    is_classif=False,
)


if __name__ == '__main__':
    unittest.main()

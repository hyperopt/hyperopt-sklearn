import unittest

from hpsklearn import \
    one_vs_rest_classifier, \
    one_vs_one_classifier, \
    output_code_classifier
from tests.utils import \
    StandardClassifierTest, \
    generate_attributes


class OneVsRestClassifierTest(StandardClassifierTest):
    """
    Class for _multiclass classification testing
    """


generate_attributes(
    TestClass=OneVsRestClassifierTest,
    fn_list=[one_vs_rest_classifier, one_vs_one_classifier, output_code_classifier],
    is_classif=True,
)


if __name__ == '__main__':
    unittest.main()

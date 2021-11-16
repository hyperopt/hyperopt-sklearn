import unittest

from hpsklearn import elliptic_envelope
from hpsklearn.tests.utils import \
    IrisTest, \
    generate_test_attributes


class TestEllipticEnvelopeClassifier(IrisTest):
    """
    Class for _elliptic_envelope classification testing
    """


generate_test_attributes(
    TestClass=TestEllipticEnvelopeClassifier,
    fn_list=[elliptic_envelope],
    is_classif=True,
)


if __name__ == '__main__':
    unittest.main()
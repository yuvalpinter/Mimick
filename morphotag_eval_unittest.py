'''
Small test to ensure proper behaviour of evaluate_morphotags.Evaluator
'''
from __future__ import division
import unittest
from evaluate_morphotags import Evaluator
from utils import split_tagstring
from numpy import nan
from numpy.testing.utils import assert_almost_equal
from numpy.testing.utils import assert_equal

__author__ = "Yuval Pinter, 2016"

class Test(unittest.TestCase):

    def testEval(self):
        eval1 = Evaluator(m = 'att')
        eval2 = Evaluator(m = 'att_val')
        eval3 = Evaluator(m = 'exact')

        with open('simple_morpho_eval_test.txt', 'r') as sample_file:
            for l in sample_file.readlines():
                if not l.startswith('#'):
                    g, o = map(split_tagstring, l.split('\t'))
                    eval1.add_instance(g, o)
                    eval2.add_instance(g, o)
                    eval3.add_instance(g, o)

        assert_almost_equal(eval1.mic_f1(), 5/9)
        assert_almost_equal(eval1.mac_f1(), 13/30)
        assert_almost_equal(eval1.mic_f1(att = "Number"), 2/5)
        assert_equal(eval1.mac_f1(att = "Number"), nan)
        assert_almost_equal(eval2.mic_f1(), 5/9)
        assert_almost_equal(eval2.mac_f1(), 29/70)
        assert_almost_equal(eval2.mac_f1(att = "Number"), 1/4)
        assert_almost_equal(eval3.acc(), 1/4)


if __name__ == "__main__":
    unittest.main()

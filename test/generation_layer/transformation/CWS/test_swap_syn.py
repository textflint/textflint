import unittest

from TextFlint.generation_layer.transformation.CWS.swap_syn import SwapSyn
from TextFlint.input_layer.component.sample.cws_sample import CWSSample

sent1 = '大哥过奖了'
data_sample = CWSSample({'x': sent1, 'y': ['B', 'E', 'B', 'E', 'S']})
swap_ins = SwapSyn()


class TestSynonym(unittest.TestCase):
    def test_transformation(self):
        trans_samples = swap_ins.transform(data_sample, n=5)
        self.assertEqual(5, len(trans_samples))
        for sample in trans_samples:
            self.assertEqual([2, 2, 2, 2, 0], sample.mask)
            self.assertEqual(['B', 'E', 'B', 'E', 'S'], sample.get_value('y'))
            self.assertTrue(sample.get_value('x')[:2] in
                            swap_ins.synonym_dict[sent1[:2]])
            self.assertTrue(sample.get_value('x')[2:4] in
                            swap_ins.synonym_dict[sent1[2:4]])


if __name__ == "__main__":
    unittest.main()

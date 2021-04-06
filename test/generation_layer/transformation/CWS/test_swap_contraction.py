import unittest

from textflint.generation_layer.transformation.CWS.swap_contraction \
    import SwapContraction
from textflint.input_layer.component.sample.cws_sample import CWSSample

sent1 = '来自 央视 报道 。'
data_sample = CWSSample({'x': sent1, 'y': []})
swap_ins = SwapContraction()


class TestSwapContraction(unittest.TestCase):
    def test_get_transformations(self):
        self.assertTrue(([[2, 4]], [['中央电视台']], [['B', 'M', 'M', 'M', 'E']]) ==
                        swap_ins._get_transformations(data_sample.get_words()))
        self.assertRaises(AssertionError,  swap_ins._get_transformations, sent1)
        self.assertRaises(AssertionError, swap_ins._get_transformations, '')
        self.assertTrue(swap_ins._get_transformations([]) == ([], [], []))

    def test_transformation(self):
        trans_sample = swap_ins.transform(data_sample)
        self.assertTrue(len(trans_sample) == 1)
        trans_sample = trans_sample[0]
        self.assertEqual([0, 0, 2, 2, 2, 2, 2, 0, 0, 0], trans_sample.mask)
        self.assertEqual('来自中央电视台报道。', trans_sample.get_value('x'))
        self.assertEqual(['B', 'E', 'B', 'M', 'M', 'M', 'E', 'B', 'E', 'S'],
                         trans_sample.get_value('y'))


if __name__ == "__main__":
    unittest.main()

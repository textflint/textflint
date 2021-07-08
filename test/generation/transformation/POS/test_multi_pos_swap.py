import unittest

from textflint.generation.transformation.POS.multi_pos_swap \
    import SwapMultiPOS
from textflint.input.component.sample.pos_sample import *

data = {'x': ['That', 'is', 'a', 'pretty', 'prefixed', 'survey'],
        'y': ['DT', 'VBZ', 'DT', 'RB', 'JJ', 'NN']}

data_sample = POSSample(data)


class TestMultiPOSSwap(unittest.TestCase):

    def test_MultiPOSSwap(self):
        self.assertRaises(AssertionError, SwapMultiPOS, 'DT')

        # test data with no words to replace
        data = POSSample({'x': ['that'] * 3, 'y': ['DT'] * 3})
        swap_ins = SwapMultiPOS('NN')
        self.assertEqual([], swap_ins.transform(data))

        # test faction
        swap_ins = SwapMultiPOS('NN')
        change = swap_ins.transform(data_sample, n=3)
        self.assertTrue(3 == len(change))
        for item in change:
            for ori_word, trans_word, mask in zip(data_sample.get_words('x'),
                                                  item.get_words('x'),
                                                  item.get_mask('x')):
                self.assertTrue(ori_word == trans_word if mask == 0
                                else ori_word != trans_word)


if __name__ == "__main__":
    unittest.main()

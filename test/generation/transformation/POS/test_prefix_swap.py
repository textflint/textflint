import unittest

from textflint.generation.transformation.POS.prefix_swap import SwapPrefix
from textflint.input.component.sample.pos_sample import *

data = {'x': ['That', 'is', 'a', 'prefixed', 'survey'],
        'y': ['DT', 'VBZ', 'DT', 'JJ', 'NN']}

data_sample = POSSample(data)
swap_ins = SwapPrefix()


class TestPrefixSwap(unittest.TestCase):
    def test_PrefixSwap(self):

        # test data with no words with prefix or postfix
        data = POSSample({'x': ['is'] * 3, 'y': ['O'] * 3})
        self.assertEqual([], swap_ins.transform(data))

        # test faction
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

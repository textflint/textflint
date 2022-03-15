import unittest

from textflint.input.component.sample.smcn_sample import SMCNSample
from textflint.generation.transformation.SMCN import SwapWord

sample1 = SMCNSample({'sentence1': '我喜欢这本书。',
        'sentence2': '这本书是我喜欢的。',
        'y': '1'})

smcn_antonymswap = SwapWord()


class TestSwapWord(unittest.TestCase):
    def test_whether_changed(self):
        # test whether the sample changed or not
        trans = smcn_antonymswap.transform(sample1)
        self.assertTrue(sample1.sentence1.field_value !=
                        trans[0].sentence1.field_value or
                        sample1.sentence2.field_value !=
                        trans[0].sentence2.field_value)

    def test_label(self):
        # SmAntonymSwap will change some word to its opposite
        # meaning, which leads y to '0'
        trans = smcn_antonymswap.transform(sample1)
        self.assertEqual('0', trans[0].y.field_value)

    def test_empty_sample(self):
        # test sample with empty string and empty list
        self.assertRaises(AttributeError, smcn_antonymswap.transform, '')
        self.assertRaises(AttributeError, smcn_antonymswap.transform, [])


if __name__ == "__main__":
    unittest.main()

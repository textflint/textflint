import unittest

from textflint.input.component.sample.sm_sample import SMSample
from textflint.generation.transformation.SM import SwapWord

sample1 = SMSample({'sentence1': 'There are two little boys smiling.',
        'sentence2': 'Two little boys are smiling and laughing while one is '
                     'standing and one is in a bouncy seat',
        'y': '1'})

sm_antonymswap = SwapWord()


class TestSwapWord(unittest.TestCase):
    def test_whether_changed(self):
        # test whether the sample changed or not
        trans = sm_antonymswap.transform(sample1)
        self.assertTrue(sample1.sentence1.field_value !=
                        trans[0].sentence1.field_value or
                        sample1.sentence2.field_value !=
                        trans[0].sentence2.field_value)

    def test_label(self):
        # SmAntonymSwap will change some word to its opposite
        # meaning, which leads y to '0'
        trans = sm_antonymswap.transform(sample1)
        self.assertEqual('0', trans[0].y.field_value)

    def test_empty_sample(self):
        # test sample with empty string and empty list
        self.assertRaises(AttributeError, sm_antonymswap.transform, '')
        self.assertRaises(AttributeError, sm_antonymswap.transform, [])


if __name__ == "__main__":
    unittest.main()

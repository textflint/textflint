import unittest

from textflint.input.component.sample.smcn_sample import SMCNSample
from textflint.generation.transformation.SMCN import SwapNum

sample1 = SMCNSample({'sentence1': '天气预报说上海今天下雨的概率是80%。',
        'sentence2': '根据天气预报，上海今天有80%的概率下雨。',
        'y': '1'})

smcn_numword = SwapNum()


class TestSwapNum(unittest.TestCase):
    def test_whether_changed(self):
        # test whether the sample changed or not
        trans = smcn_numword.transform(sample1)
        if len(trans)>0:
            self.assertTrue(sample1.sentence1.field_value !=
                            trans[0].sentence1.field_value or
                            sample1.sentence2.field_value !=
                            trans[0].sentence2.field_value)

    def test_label(self):
        # SmNumWord will change some thing about number,
        # which leads contradiction label
        trans = smcn_numword.transform(sample1)
        if len(trans)>0:
            self.assertEqual('0', trans[0].y.field_value)

    def test_empty_sample(self):
        # test sample with empty string and empty list
        self.assertRaises(AttributeError, smcn_numword.transform, '')
        self.assertRaises(AttributeError, smcn_numword.transform, [])


if __name__ == "__main__":
    unittest.main()

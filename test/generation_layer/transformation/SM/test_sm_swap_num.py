import unittest

from textflint.input_layer.component.sample.sm_sample import SMSample
from textflint.generation_layer.transformation.SM import SwapNum

sample1 = SMSample({'sentence1': 'MR zhang has 10 students',
        'sentence2': 'Mr zhang has 10 students',
        'y': '1'})

sm_numword = SwapNum()


class TestSwapNum(unittest.TestCase):
    def test_whether_changed(self):
        # test whether the sample changed or not
        trans = sm_numword.transform(sample1)
        self.assertTrue(sample1.sentence1.field_value !=
                        trans[0].sentence1.field_value or
                        sample1.sentence2.field_value !=
                        trans[0].sentence2.field_value)

    def test_label(self):
        # SmNumWord will change some thing about number,
        # which leads contradiction label
        trans = sm_numword.transform(sample1)
        self.assertEqual('0', trans[0].y.field_value)

    def test_empty_sample(self):
        # test sample with empty string and empty list
        self.assertRaises(AttributeError, sm_numword.transform, '')
        self.assertRaises(AttributeError, sm_numword.transform, [])


if __name__ == "__main__":
    unittest.main()

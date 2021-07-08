import unittest

from textflint.input.component.sample.re_sample import RESample
from textflint.generation.transformation.RE.swap_ent import SwapEnt

data = {'x': ["``", "The", "situation", "is", "very", "serious", ",", "''",
              "Mattis", ",", "30", ",", "told", "reporters", "after", "meeting",
              "with", "Ban", "in", "New", "York", "."],
        'subj': [8, 8], 'obj': [10, 10], 'y': 'age'}
sample = SwapEnt('lowfreq')
re_data = RESample(data)


class TestEntitySwap(unittest.TestCase):
    def test_replace_en(self):
        self.assertRaises(AssertionError, sample.replace_en,
                          [], [8, 8], data['x'])
        self.assertRaises(AssertionError, sample.replace_en,
                          'PERSON', [], data['x'])
        self.assertRaises(AssertionError, sample.replace_en,
                          'PERSON', [8, 8], [])
        token, length = sample.replace_en('PERSON', [8, 8], data['x'])
        self.assertTrue(token != data['x'])
        self.assertTrue(isinstance(length, int))

    def test_subj_and_obj_transform(self):
        self.assertRaises(AssertionError, sample.subj_and_obj_transform,
                          re_data, [], [1, 2, 3, 4, 'PERSON', 'PERSON'])
        self.assertRaises(AssertionError, sample.subj_and_obj_transform,
                          [], 1, [1, 2, 3, 4, 'PERSON', 'PERSON'])
        self.assertRaises(AssertionError, sample.subj_and_obj_transform,
                          re_data, 1, [1, 2, 3, 4, 'PERSON', 'None'])
        trans_samples = sample.subj_and_obj_transform(
            re_data, 1, [8, 8, 10, 10, 'PERSON', 'PERSON'])
        for trans_sample in trans_samples:
            sent, label = trans_sample.get_sent()
            self.assertTrue(sent!=data['x'])

    def test_single_transform(self):
        self.assertRaises(AssertionError, sample.single_transform,
                          re_data, [], [1, 2, 3, 4, 'PERSON', 'None'])
        self.assertRaises(AssertionError, sample.single_transform,
                          [], 1, [1, 2, 3, 4, 'PERSON', 'None'])
        self.assertRaises(AssertionError, sample.single_transform,
                          re_data, 1, [3, 2, 3, 4, 'PERSON', 'PERSON'])
        trans_samples = sample.single_transform(re_data, 1,
                                                [8, 8, 10, 10, 'PERSON', 'None'])
        for trans_sample in trans_samples:
            sent, label = trans_sample.get_sent()
            self.assertTrue(sent != data['x'])

    def test_transform(self):
        self.assertRaises(AssertionError, sample._transform, [], 1)
        self.assertRaises(AssertionError, sample._transform, re_data, [])
        trans_samples = sample._transform(re_data, 1)
        for trans_sample in trans_samples:
            self.assertTrue(type(trans_sample) == type(re_data))


if __name__ == "__main__":
    unittest.main()

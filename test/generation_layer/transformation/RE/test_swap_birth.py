import unittest
import sys
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from textflint.input_layer.component.sample.re_sample import RESample
from textflint.generation_layer.transformation.RE.swap_birth import SwapBirth

data = {'x': ["``", "The", "situation", "is", "very", "serious", ",", "''",
              "Mattis", ",", "30", ",", "told", "reporters", "after",
              "meeting", "with", "Ban", "in", "New", "York", "."],
        'subj': [8, 8], 'obj': [10, 10], 'y': 'age'}
sample = SwapBirth()
re_data = RESample(data)


class TestBirthSwap(unittest.TestCase):

    def test_generate_new_sen_for_birth(self):
        self.assertRaises(AssertionError, sample.generate_new_sen_for_birth, 6,
                          ["a", "b", "c", "d", "e"], ["NP"] * 5, [0, 2, 3, 1, 4],
                          ["ROOT", "a", "a", "a", "a"])
        self.assertRaises(AssertionError, sample.generate_new_sen_for_birth, 2,
                          ["a", "b", "c", "d", "e"], ["NP"]*4, [0, 2, 3, 1, 4],
                          ["ROOT", "a", "a", "a", "a"])
        self.assertRaises(AssertionError, sample.generate_new_sen_for_birth, 2,
                          ["a", "b", "c", "d", "e"], ["NP"]*5, [0, 2, 3, 1, 4],
                          ["a", "a", "a", "a", "a"])
        self.assertRaises(AssertionError, sample.generate_new_sen_for_birth, 2,
                          ["a", "b", "c", "d", "e"], ["NP"]*5, [1, 2, 3, 1, 4],
                          ["ROOT", "a", "a", "a", "a"])
        self.assertRaises(AssertionError, sample.generate_new_sen_for_birth, 2,
                          "abcde", ["NP"] * 5, [0, 2, 3, 1, 4],
                          ["ROOT", "a", "a", "a", "a"])

    def test_transform(self):
        self.assertRaises(AssertionError, sample._transform, [], 1)
        self.assertRaises(AssertionError, sample._transform, re_data, [])
        trans_samples = sample._transform(re_data, 1)
        for trans_sample in trans_samples:
            self.assertTrue(type(trans_sample) == type(re_data))


if __name__ == "__main__":
    unittest.main()

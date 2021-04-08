import unittest
import sys
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from textflint.input_layer.component.sample.re_sample import RESample
from textflint.generation_layer.transformation.RE.swap_employee \
    import SwapEmployee


data = {'x': ["``", "The", "situation", "is", "very", "serious", ",", "''",
              "Mattis", ",", "30", ",", "told", "reporters", "after", "meeting",
              "with", "Ban", "in", "New", "York", "."],
        'subj': [8, 8], 'obj': [10, 10], 'y': 'age'}


class TestEmployeeSwap(unittest.TestCase):
    trans_sample = SwapEmployee()
    re_data = RESample(data)

    def test_generate_new_item(self):
        self.assertRaises(
            AssertionError, self.trans_sample.generate_new_item, True,
            ["a"], ["b"], ["c"], ["d"], ["e"], 0
        )
        self.assertRaises(
            AssertionError, self.trans_sample.generate_new_item, True,
            ["a"], 1, ["c"], ["d"], ["e"], [0, 1]
        )
        self.assertRaises(
            AssertionError, self.trans_sample.generate_new_item, True,
            ["a"], ["b"], ["c"], ["d"], ["e"], [0, 2]
        )

    def test_assert_attributive(self):
        self.assertRaises(
            AssertionError, self.trans_sample.assert_attributive,
            ['a', 'b'], 'c', ['a', 'b', 'c'], [1, 2, 0], ['c'], [1]
        )
        self.assertRaises(
            AssertionError, self.trans_sample.assert_attributive,
            ['a', 'b'], ['c'], ['a', 'b', 'c'], [1, 2, 3], ['c'], [1]
        )

    def test_split_sent(self):
        self.assertRaises(
            AssertionError, self.trans_sample.split_sent,
            [], [], ["a", "b", "c"]
        )
        self.assertRaises(
            AssertionError, self.trans_sample.split_sent,
            [1], 2, ["a", "b", "c"]
        )
        self.assertRaises(
            AssertionError, self.trans_sample.split_sent,
            [1, 0], [1, 2], ["a", "b", "c"]
        )
        self.assertRaises(
            AssertionError, self.trans_sample.split_sent,
            [-1, 0], [1, 2], ["a", "b", "c"]
        )
        self.assertRaises(
            AssertionError, self.trans_sample.split_sent,
            [0, 1], [1, 3], ["a", "b", "c"]
        )

    def test_transform(self):
        self.assertRaises(AssertionError, self.trans_sample._transform, [], 1)
        self.assertRaises(AssertionError, self.trans_sample._transform,
                          self.re_data, [])
        trans_samples = self.trans_sample._transform(self.re_data, 1)
        for trans_sample in trans_samples:
            self.assertTrue(type(trans_sample)==type(self.re_data))


if __name__ == "__main__":
    unittest.main()

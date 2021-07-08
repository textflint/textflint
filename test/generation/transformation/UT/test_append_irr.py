import unittest
import random

from textflint.input.component.sample import SASample
from textflint.generation.transformation.UT.append_irr import *

sent1 = 'The quick brown fox jumps over the lazy dog.'
data_sample = SASample({'x': sent1, 'y': "negative"})
swap_ins = AppendIrr()


class TestAppendIrr(unittest.TestCase):
    def test_transformation(self):
        # test the change num
        change_sample = swap_ins.transform(data_sample, n=5)
        self.assertEqual(5, len(change_sample))

        # test if the item change
        begin = []
        end = []
        for sample in change_sample:
            self.assertTrue(sent1 in sample.get_text('x'))
            index = sample.get_text('x').index(sent1)
            begin.append(sample.get_text('x')[:index])
            end.append(sample.get_text('x')[index + len(sent1):])

        # test if the n change samples not equal
        self.assertEqual(len(set(begin)), 5)
        self.assertEqual(len(set(end)), 5)

        random.seed(100)
        special_sample = SASample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        special_sample = swap_ins.transform(special_sample)[0]
        self.assertEqual('I actually wanted to talk to you, ~!@#$%^7890 "\'',
                         special_sample.get_text('x'))
        # TODO repair bug 空串
        # special_sample = SASample({'x': '', 'y': "negative"})
        # print(swap_ins.transform(special_sample))


if __name__ == "__main__":
    unittest.main()

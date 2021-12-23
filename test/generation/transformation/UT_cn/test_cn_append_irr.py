import unittest
import random

from textflint.input.component.sample import UTCnSample
from textflint.generation.transformation.UT_cn.cn_append_irr import *

sent1 = '那只敏捷的棕色狐狸跳过了那只懒惰的狗。'
data_sample = UTCnSample({'x': sent1, 'y': "negative"})
trans_method = AppendIrr()


class TestAppendIrr(unittest.TestCase):
    def test_transformation(self):
        # test the change num
        change_sample = trans_method.transform(data_sample, n=5)
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
        special_sample = UTCnSample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        special_sample = trans_method.transform(special_sample)[0]
        self.assertEqual('众所周知，~!@#$%^7890"\'哭声多，羊毛少。',special_sample.get_text('x'))

        special_sample = UTCnSample({'x': '', 'y': "negative"})
        special_sample = trans_method.transform(special_sample)[0]
        self.assertEqual('此外，', special_sample.get_text('x'))

if __name__ == "__main__":
    unittest.main()

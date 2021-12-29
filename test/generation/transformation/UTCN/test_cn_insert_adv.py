import unittest

import random
from textflint.input.component.sample import UTCnSample
from textflint.generation.transformation.UTCN import InsertAdv

random.seed(100)
sent1 = '那只敏捷的棕色狐狸跳过了那只懒惰的狗。'
data_sample = UTCnSample({'x': sent1, 'y': "negative"})
trans_method = InsertAdv()


class TestAddAdverb(unittest.TestCase):
    def test_transformation(self):
        # test the change num
        change_sample = trans_method.transform(data_sample, n=5)
        self.assertEqual(5, len(change_sample))

        special_sample = UTCnSample({'x': '', 'y': "negative"})
        self.assertEqual([], trans_method.transform(special_sample))
        special_sample = UTCnSample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        self.assertEqual([], trans_method.transform(special_sample))

        # test if the item change
        change_item = []
        for sample in change_sample:
            self.assertEqual(data_sample.get_tokens('x')[:9],
                             sample.get_tokens('x')[:9])
            self.assertEqual(data_sample.get_tokens('x')[-10:],
                             sample.get_tokens('x')[-10:])


if __name__ == "__main__":
    unittest.main()

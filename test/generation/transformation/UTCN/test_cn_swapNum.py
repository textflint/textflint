import random
import unittest

from textflint.input.component.sample import UTCnSample
from textflint.generation.transformation.UTCN import CnSwapNum

sample = UTCnSample({
    'x': '今天5个人聚餐。',
    'y': 1,
})
trans_method = CnSwapNum()


class TestSwapNum(unittest.TestCase):
    def test_transformation(self):
        special_sample = UTCnSample({'x': '', 'y': "negative"})
        self.assertEqual([], trans_method.transform(special_sample))

        special_sample = UTCnSample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        self.assertEqual([], trans_method.transform(special_sample))

        random.seed(100)
        # # test if the item change
        change_sample = trans_method.transform(sample, n=5)
        self.assertTrue(5 >= len(change_sample))
        for s in change_sample:
            self.assertEqual(sample.get_words('x')[:1], s.get_words('x')[:1])
            self.assertEqual(sample.get_words('x')[-4:], s.get_words('x')[-4:])

        self.assertEqual(change_sample[0].get_value('x'), '今天4个人聚餐。')


if __name__ == "__main__":
    unittest.main()

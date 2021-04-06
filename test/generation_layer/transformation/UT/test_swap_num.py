import unittest
import editdistance

from TextFlint.input_layer.component.sample import SASample
from TextFlint.generation_layer.transformation.UT.swap_num import SwapNum


class TestNumber(unittest.TestCase):
    def test_transformation(self):
        import random
        random.seed(100)
        sent1 = "Here's 0 bug."
        data_sample = SASample({'x': sent1, 'y': "negative"})
        swap_ins = SwapNum()
        x = swap_ins.transform(data_sample, n=3)

        self.assertTrue(3 == len(x))
        for sample in x:
            self.assertTrue(editdistance.distance(sample.get_text('x'),
                                                  sent1) == 1)
            self.assertTrue("negative" == sample.get_value('y'))

        self.assertTrue(x[0].get_text('x') == "Here's 1 bug.")

        special_sample = SASample({'x': '', 'y': "negative"})
        self.assertEqual([], swap_ins.transform(special_sample))
        special_sample = SASample({'x': '~!@#$%^"\'', 'y': "negative"})
        self.assertEqual([], swap_ins.transform(special_sample))


if __name__ == "__main__":
    unittest.main()

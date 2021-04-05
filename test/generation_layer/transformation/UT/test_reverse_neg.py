import unittest

from TextFlint.input_layer.component.sample import SASample
from TextFlint.generation_layer.transformation.UT.reverse_neg import ReverseNeg


class TestAddAdverb(unittest.TestCase):
    def test_transformation(self):
        sent1 = "The quick brown fox jumps over the lazy dog ."
        data_sample = SASample({'x': sent1, 'y': "negative"})
        swap_ins = ReverseNeg()
        x = swap_ins.transform(data_sample, n=1)
        x = x[0]
        self.assertEqual('The quick brown fox does not jumps over the lazy '
                         'dog.', x.get_text('x'))
        self.assertEqual('negative', x.get_value('y'))

        # test special input
        special_sample = SASample({'x': '', 'y': "negative"})
        swap_ins.transform(special_sample)
        self.assertEqual([], swap_ins.transform(special_sample))
        special_sample = SASample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        self.assertEqual([], swap_ins.transform(special_sample))


if __name__ == "__main__":
    unittest.main()

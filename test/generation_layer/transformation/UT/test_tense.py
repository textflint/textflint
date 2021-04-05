import unittest

from TextFlint.input_layer.component.sample import SASample
from TextFlint.generation_layer.transformation.UT.tense import Tense


class TestTense(unittest.TestCase):
    def test_transformation(self):

        sent1 = 'The quick brown fox jumps over the lazy dog .'
        data_sample = SASample({'x': sent1, 'y': "negative"})
        swap_ins = Tense()
        x = swap_ins.transform(data_sample, n=3)

        self.assertTrue(3 == len(x))
        change = []
        for sample in x:
            origin = data_sample.get_words('x')
            trans = sample.get_words('x')
            self.assertEqual(origin[:4], trans[:4])
            self.assertEqual(origin[5:], trans[5:])
            change.append(trans[4])
            self.assertTrue(trans[4] != origin[4])

        # test special input
        special_sample = SASample({'x': '', 'y': "negative"})
        self.assertEqual([], swap_ins.transform(special_sample))
        special_sample = SASample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        self.assertEqual([], swap_ins.transform(special_sample))


if __name__ == "__main__":
    unittest.main()

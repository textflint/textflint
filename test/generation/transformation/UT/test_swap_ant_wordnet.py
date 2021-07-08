import unittest

from textflint.input.component.sample import SASample
from textflint.generation.transformation.UT.swap_ant_wordnet \
    import SwapAntWordNet


class TestSwapAntWordNet(unittest.TestCase):
    def test_transformation(self):
        sent1 = 'The fast brown fox jumps over the lazy dog .'
        data_sample = SASample({'x': sent1, 'y': "negative"})
        swap_ins = SwapAntWordNet()
        x = swap_ins.transform(data_sample, n=5)
        self.assertTrue(1 == len(x))

        for sample in x:
            origin = data_sample.get_words('x')
            trans = sample.get_words('x')
            self.assertEqual(origin[0], trans[0])
            self.assertTrue(origin[1] != trans[1])
            self.assertEqual(origin[2:], trans[2:])

        special_sample = SASample({'x': '', 'y': "negative"})
        self.assertEqual([], swap_ins.transform(special_sample))
        special_sample = SASample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        self.assertEqual([], swap_ins.transform(special_sample))


if __name__ == "__main__":
    unittest.main()

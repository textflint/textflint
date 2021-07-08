import unittest

from textflint.input.component.sample import SASample
from textflint.generation.transformation.UT.spelling_error \
    import SpellingError


class TestSpellingError(unittest.TestCase):
    def test_transformation(self):
        import random
        random.seed(100)
        sent1 = 'The quick brown fox jumps over the lazy dog.'
        data_sample = SASample({'x': sent1, 'y': "negative"})
        swap_ins = SpellingError()
        x = swap_ins.transform(data_sample, n=5)

        change = []
        for sample in x:
            origin = data_sample.get_words('x')
            trans = sample.get_words('x')
            self.assertEqual(origin[0], trans[0])
            self.assertEqual(origin[2:7], trans[2:7])
            change.append(trans[1] + trans[7] + trans[8])
            self.assertTrue(trans[1] != origin[1])
            self.assertTrue(trans[7] != origin[7])
            self.assertTrue(trans[8] != origin[8])

        self.assertTrue(5 == len(set(change)))

        special_sample = SASample({'x': '', 'y': "negative"})
        self.assertEqual([], swap_ins.transform(special_sample))
        special_sample = SASample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        self.assertEqual([], swap_ins.transform(special_sample))


if __name__ == "__main__":
    unittest.main()

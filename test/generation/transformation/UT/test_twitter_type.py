import unittest

from textflint.input.component.sample import SASample
from textflint.generation.transformation.UT.twitter_type \
    import TwitterType


class TestTwitterType(unittest.TestCase):
    def test_transformation(self):
        sent1 = "The quick brown fox jumps over the lazy dog."
        data_sample = SASample({'x': sent1, 'y': "negative"})
        swap_ins = TwitterType()
        x = swap_ins.transform(data_sample, n=5)

        self.assertEqual(5, len(x))
        change = []
        for sample in x:
            origin = data_sample.get_words('x')
            trans = sample.get_words('x')
            change.append(sample.get_text('x'))
            self.assertTrue(origin == trans[1:] or origin == trans[:-1])

        self.assertTrue(5 == len(set(x)))

        # test special input
        special_sample = SASample({'x': '', 'y': "negative"})
        x = swap_ins.transform(special_sample)
        self.assertEqual(1, len(x))
        x = x[0]
        self.assertTrue(x.get_text('x') != '')
        special_sample = SASample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        x = swap_ins.transform(special_sample)
        self.assertEqual(1, len(x))
        x = x[0]
        self.assertTrue(x.get_text('x') != '~!@#$%^7890"\'')


if __name__ == "__main__":
    unittest.main()

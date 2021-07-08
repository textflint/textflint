import unittest

from textflint.input.component.sample import SASample
from textflint.generation.transformation.UT.punctuation import Punctuation


class TestPunctuation(unittest.TestCase):
    def test_transformation(self):

        sent1 = "The quick brown fox jumps over the lazy dog."
        data_sample = SASample({'x': sent1, 'y': "negative"})
        swap_ins = Punctuation()
        x = swap_ins.transform(data_sample, n=10)

        # test the form
        for _sample in x:
            self.assertEqual(''.join(filter(str.isalpha, _sample.get_text('x'))), ''.join(filter(str.isalpha, data_sample.get_text('x'))))
            self.assertTrue(_sample.get_text('x') != data_sample.get_text('x'))

        self.assertTrue(10 >= len(x))

        import random
        random.seed(100)
        x = swap_ins.transform(data_sample, n=1)
        x = x[0]
        # test the data
        self.assertTrue('{ The quick brown fox jumps over the lazy dog; }' == x.get_text('x'))

        special_sample = SASample({'x': '', 'y': "negative"})
        self.assertEqual(1, len(swap_ins.transform(special_sample)))
        special_sample = SASample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        self.assertEqual(1, len(swap_ins.transform(special_sample)))


if __name__ == "__main__":
    unittest.main()

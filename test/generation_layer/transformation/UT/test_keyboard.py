import unittest

from textflint.input_layer.component.sample import SASample
from textflint.generation_layer.transformation.UT.keyboard import Keyboard


class TestKeyboard(unittest.TestCase):
    def test_transformation(self):
        sent1 = 'The quick brown fox jumps over the lazy dog .'
        data_sample = SASample({'x': sent1, 'y': "negative"})
        trans = Keyboard()

        import random
        random.seed(173)

        x = trans.transform(data_sample, n=5)
        self.assertEqual(5, len(x))

        sents = ["The quick brown f8x jumps over the lzzy dog.",
                 "The quick brown fpx jumps over the lXzy dog.",
                 "The quick brown box jumps over the lzzy dog.",
                 "The quick brown foC jumps over the lXzy dog.",
                 "The quick brown foS jumps over the lazg dog."]
        for sample, sent in zip(x, sents):
            self.assertEqual(sample.get_text('x'), sent)

        # test special data
        special_sample = SASample({'x': '', 'y': "negative"})
        self.assertEqual([], trans.transform(special_sample))
        special_sample = SASample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        special_sample = trans.transform(special_sample)
        self.assertEqual("~!w#$%^7890 \"'", special_sample[0].get_text('x'))


if __name__ == "__main__":
    unittest.main()

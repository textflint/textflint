import unittest

from textflint.input_layer.component.sample import SASample
from textflint.generation_layer.transformation.UT.ocr \
    import Ocr


class TestOCR(unittest.TestCase):
    def test_transformation(self):
        sent2 = 'The quick brown fox jumps over the lazy dog. '
        data_sample = SASample({'x': sent2, 'y': "negative"})
        import random
        random.seed(100)
        swap_ins = Ocr()

        x = swap_ins.transform(data_sample, n=5)
        self.assertEqual(4, len(x))

        self.assertTrue(x[0].get_words('x')[1] == 'qoick')
        self.assertTrue(x[1].get_words('x')[1] == 'quicr')
        self.assertTrue(x[2].get_words('x')[3] == 'fux')
        self.assertTrue(x[3].get_words('x')[1] == 'quicr')

        special_sample = SASample({'x': '', 'y': "negative"})
        self.assertEqual([], swap_ins.transform(special_sample))
        special_sample = SASample({'x': '~!@#$%^7890g"\'', 'y': "negative"})
        special_sample = swap_ins.transform(special_sample)[0]
        self.assertEqual('~!@#$%^789O g "\'', special_sample.get_text('x'))


if __name__ == "__main__":
    unittest.main()

import unittest

from textflint.input.component.sample import SASample
from textflint.generation.transformation.UT.mlm_suggestion \
    import MLMSuggestion


class TestMLM(unittest.TestCase):
    @unittest.skip("Manual test")
    def test_transformation(self):
        sent2 = 'The quick brown fox jumps over the lazy dog. '
        data_sample = SASample({'x': sent2, 'y': "negative"})
        import random
        random.seed(100)
        swap_ins = MLMSuggestion(device='cpu')

        x = swap_ins.transform(data_sample, n=5)
        self.assertEqual(5, len(x))
        tokens = []
        for _sample in x:
            self.assertTrue(_sample.get_words('x')[:2] == data_sample.get_words('x')[:2])
            self.assertTrue(_sample.get_words('x')[3] == data_sample.get_words('x')[3])
            self.assertTrue(_sample.get_words('x')[-5:] == data_sample.get_words('x')[-5:])
            tokens.append(_sample.get_words('x')[2] + _sample.get_words('x')[4])

        self.assertTrue(5 == len(set(tokens)))

        special_sample = SASample({'x': '', 'y': "negative"})
        self.assertEqual([], swap_ins.transform(special_sample))
        special_sample = SASample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        special_sample = swap_ins.transform(special_sample)[0]
        self.assertEqual('epilogue "\'', special_sample.get_text('x'))


if __name__ == "__main__":
    unittest.main()

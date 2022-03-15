import unittest

from textflint.input.component.sample import UTCnSample
from textflint.generation.transformation.UTCN   import MLMSuggestion


class TestMLM(unittest.TestCase):
    @unittest.skip("Manual test")
    def test_transformation(self):
        sent2 = '今天天气不错。'
        data_sample = UTCnSample({'x': sent2, 'y': "negative"})
        import random
        random.seed(100)
        trans_method = MLMSuggestion(device='cpu')

        x = trans_method.transform(data_sample, n=5)
        transformation_results = ['当时天气不错。','周日天气不错。','今里天气不错。','那季天气不错。','冬年天气不错。']

        self.assertEqual(5, len(x))
        for index, _sample in enumerate(x):
            self.assertTrue(transformation_results[index] == _sample.x.text)

        special_sample = UTCnSample({'x': '', 'y': "negative"})
        self.assertEqual([], trans_method.transform(special_sample))


if __name__ == "__main__":
    unittest.main()

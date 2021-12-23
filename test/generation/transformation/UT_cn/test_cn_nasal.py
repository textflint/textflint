import unittest

from textflint.input.component.sample import UTCnSample
from textflint.generation.transformation.UT_cn \
    import CnNasal


class TestCnSwapSynWordEmbedding(unittest.TestCase):
    def test_transformation(self):
        import random
        random.seed(100)
        sent1 = '昂嗯按宁您能嫩黄换'
        data_sample = UTCnSample({'x': sent1, 'y': "negative"})
        trans_method = CnNasal()
        x = trans_method.transform(data_sample, n=5)
        self.assertTrue(5 == len(x))

        transformation_results = ['昂嗯按您您能嫩黄换','昂嗯按囜您能嫩黄换','昂嗯按脌您能嫩黄换','昂嗯按拰您能嫩黄换','昂嗯按恁您能嫩黄换']

        for index, _sample in enumerate(x):
            self.assertTrue(transformation_results[index] == _sample.x.text)

        special_sample = UTCnSample({'x': '', 'y': "negative"})
        self.assertEqual([], trans_method.transform(special_sample))
        special_sample = UTCnSample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        self.assertEqual([], trans_method.transform(special_sample))


if __name__ == "__main__":
    unittest.main()

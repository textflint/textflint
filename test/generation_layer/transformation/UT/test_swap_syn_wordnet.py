import unittest

from TextFlint.input_layer.component.sample import SASample
from TextFlint.generation_layer.transformation.UT.swap_syn_wordnet \
    import SwapSynWordNet


class TestSwapSynWordNet(unittest.TestCase):
    def test_transformation(self):
        import random
        random.seed(1)
        sent1 = 'The quick brown fox jumps over the lazy dog .'
        data_sample = SASample({'x': sent1, 'y': "negative"})
        swap_ins = SwapSynWordNet()
        x = swap_ins.transform(data_sample, n=5)

        self.assertTrue(5 == len(x))
        for sample in x:
            cnt = 0
            for i, j in zip(sample.get_words('x'), data_sample.get_words('x')):
                if i != j:
                    cnt += 1
            self.assertTrue(cnt == 1)

        special_sample = SASample({'x': '', 'y': "negative"})
        self.assertEqual([], swap_ins.transform(special_sample))
        special_sample = SASample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        self.assertEqual([], swap_ins.transform(special_sample))


if __name__ == "__main__":
    unittest.main()

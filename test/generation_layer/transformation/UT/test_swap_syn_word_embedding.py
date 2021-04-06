import unittest

from textflint.input_layer.component.sample import SASample
from textflint.generation_layer.transformation.UT.swap_syn_word_embedding \
    import SwapSynWordEmbedding


class TestSwapSynWordEmbedding(unittest.TestCase):
    def test_transformation(self):
        import random
        random.seed(1)
        sent1 = "There are no water in bottom."
        data_sample = SASample({'x': sent1, 'y': "negative"})
        swap_ins = SwapSynWordEmbedding()
        x = swap_ins.transform(data_sample, n=5)
        self.assertTrue(5 == len(x))

        change = []
        for sample in x:
            origin = data_sample.get_words('x')
            trans = sample.get_words('x')
            change.append(trans[-2])
            self.assertEqual(origin[:5], trans[:5])
            self.assertEqual(origin[-1], trans[-1])
        self.assertEqual(5, len(set(change)))

        special_sample = SASample({'x': '', 'y': "negative"})
        self.assertEqual([], swap_ins.transform(special_sample))
        special_sample = SASample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        self.assertEqual([], swap_ins.transform(special_sample))


if __name__ == "__main__":
    unittest.main()

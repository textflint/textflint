import unittest

from textflint.generation_layer.transformation.CWS.swap_verb import SwapVerb
from textflint.input_layer.component.sample.cws_sample import CWSSample

sent1 = '小明想看书'
sample = CWSSample({'x': sent1, 'y': ['B', 'E', 'S', 'B', 'E']})
swap_ins = SwapVerb()


class TestSwapVerb(unittest.TestCase):
    def test_transformation(self):
        trans_sample = swap_ins.transform(sample)
        self.assertTrue(1 == len(trans_sample))
        trans_sample = trans_sample[0]
        self.assertEqual('小明想看一看书', trans_sample.get_value('x'))
        self.assertEqual(['B', 'E', 'S', 'B', 'M', 'M', 'E'],
                         trans_sample.get_value('y'))
        self.assertEqual([0, 0, 0, 2, 2, 2, 0], trans_sample.mask)


if __name__ == "__main__":
    unittest.main()

import unittest
import editdistance

from textflint.input.component.sample import NMTSample
from textflint.generation.transformation.NMT.swap_parallel_num import SwapParallelNum


class TestSwapParallelNumber(unittest.TestCase):
    def test_transformation(self):
        import random
        random.seed(100)
        source = "The last group on Friday evening departs at midnight, allowing riders 42 hours to complete the race."
        target = "Der letzten Freitags-Gruppe mit Startzeit um 24.00 Uhr stehen 42 Stunden zur Verfügung."
        data_sample = NMTSample({'source': source, 'target': target})
        swap_ins = SwapParallelNum()
        x = swap_ins.transform(data_sample, n=3, field=['source', 'target'])
        self.assertEqual(3, len(x))
        for sample in x:
            source_origin = data_sample.get_words('source')
            source_trans = sample.get_words('source')
            target_origin = data_sample.get_words('target')
            target_trans = sample.get_words('target')
            self.assertTrue(len(source_origin) == len(source_trans))
            self.assertTrue(len(target_origin) == len(target_trans))


if __name__ == "__main__":
    unittest.main()

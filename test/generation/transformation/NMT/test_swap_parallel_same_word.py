import unittest
import editdistance

from textflint.input.component.sample import NMTSample
from textflint.generation.transformation.NMT.swap_parallel_same_word import SwapParallelSameWord


class TestSwapParallelSameWord(unittest.TestCase):
    def test_transformation(self):
        source = "This star-forming region - rather unromantically named W5 by scientists - was discovered by the Spitzer telescope in the Cassiopeia constellation, at a distance of 6,500 light years away."
        target = "Diese sternenbildende Region - von Wissenschaftlern unromantisch W5 genannt - hat das Teleskop Spitzer im Sternenbild Cassiopeia entdeckt, in einer Entfernung von 6500 Lichtjahren."
        data_sample = NMTSample({'source': source, 'target': target})
        swap_ins = SwapParallelSameWord()
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

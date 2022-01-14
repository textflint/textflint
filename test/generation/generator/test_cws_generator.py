import unittest
import os
import sys
sys.path.insert(0,os.getcwd())
from textflint.input.component.sample.cws_sample import CWSSample
from textflint.input.dataset import Dataset
from textflint.generation.generator.cws_generator import CWSGenerator


class TestSpecialEntityTyposSwap(unittest.TestCase):
    def test_generate(self):
        test1 = CWSSample({'x': '', 'y': []})
        test2 = CWSSample({'x': '~ ! @ # $ % ^ & * ( ) _ +', 'y': []})
        dataset = Dataset('CWS')
        dataset.load([test1, test2])
        mode = ['SwapName',
                'CnSwapNum',
                'Reduplication',
                'CnMLM',
                'SwapContraction',
                'SwapVerb',
                'SwapSyn']
        gene = CWSGenerator(trans_methods=mode,
                            sub_methods=[])
        for original_samples, trans_rst, trans_type in gene.generate(dataset):
            self.assertTrue(len(original_samples) == 0)
            self.assertTrue(len(trans_rst) == 0)

        # test wrong trans_methods
        gene = CWSGenerator(trans_methods=["wrong_transform_method"],
                            sub_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = CWSGenerator(trans_methods=["AddSubtree"],
                            sub_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = CWSGenerator(trans_methods="CnMLM",
                            sub_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))

        sent1 = '周小明生产一万'
        sent2 = '央视想朦胧'
        dataset = Dataset(task='CWS')
        dataset.load({'x': [sent1, sent2], 'y': [
            ['B', 'M', 'E', 'B', 'E', 'B', 'E'], ['B', 'E', 'S', 'B', 'E']]})

        gene = CWSGenerator(trans_methods=mode,
                            sub_methods=[])
        for original_samples, trans_rst, trans_type in gene.generate(dataset):
            self.assertTrue(len(original_samples) == len(trans_rst))


if __name__ == "__main__":
    unittest.main()

import unittest
from textflint.input_layer.component.sample.cws_sample import CWSSample
from textflint.input_layer.dataset import Dataset
from textflint.generation_layer.generator.cws_generator import CWSGenerator


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
        gene = CWSGenerator(transformation_methods=mode,
                            subpopulation_methods=[])
        for original_samples, trans_rst, trans_type in gene.generate(dataset):
            self.assertTrue(len(original_samples) == 0)
            self.assertTrue(len(trans_rst) == 0)

        # test wrong transformation_methods
        gene = CWSGenerator(transformation_methods=["wrong_transform_method"],
                            subpopulation_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = CWSGenerator(transformation_methods=["AddSubtree"],
                            subpopulation_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = CWSGenerator(transformation_methods="CnMLM",
                            subpopulation_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))

        sent1 = '周小明生产一万'
        sent2 = '央视想朦胧'
        dataset = Dataset(task='CWS')
        dataset.load({'x': [sent1, sent2], 'y': [
            ['B', 'M', 'E', 'B', 'E', 'B', 'E'], ['B', 'E', 'S', 'B', 'E']]})

        gene = CWSGenerator(transformation_methods=mode,
                            subpopulation_methods=[])
        for original_samples, trans_rst, trans_type in gene.generate(dataset):
            self.assertTrue(len(original_samples) == len(trans_rst))


if __name__ == "__main__":
    unittest.main()

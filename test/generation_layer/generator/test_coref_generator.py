import unittest

from TextFlint.generation_layer.generator.coref_generator import CorefGenerator
from TextFlint.input_layer.dataset import Dataset
from test.data.coref_debug import CorefDebug

sample1 = CorefDebug.coref_sample1()
sample2 = CorefDebug.coref_sample2()
sample3 = CorefDebug.coref_sample3()
sample4 = CorefDebug.coref_sample4()
sample5 = CorefDebug.coref_sample5()
sample6 = CorefDebug.coref_sample6()
samples = [sample1, sample2, sample3, sample4, sample5, sample6]
dataset = Dataset("COREF")
dataset.load(samples)


class TestCorefGenerator(unittest.TestCase):

    def test_generate(self):
        gene = CorefGenerator(transformation_methods=["RndConcat"],
                              subpopulation_methods=[])
        print(gene.transformation_methods)
        for original_samples, trans_rst, trans_type in gene.generate(dataset):
            self.assertEqual(len(original_samples), len(trans_rst))  
        gene = CorefGenerator(transformation_methods=["wrong_transform_method"],
                              subpopulation_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = CorefGenerator(transformation_methods=["AddSubtree"],
                              subpopulation_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = CorefGenerator(transformation_methods="OOV",
                              subpopulation_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = CorefGenerator(transformation_methods=["Contraction", "SwapNamedEnt"],
                              subpopulation_methods=[])
        for original_samples, trans_rst, trans_type in gene.generate(dataset):
            self.assertTrue(len(original_samples) >= len(trans_rst))  


if __name__ == "__main__":
    unittest.main()

from textflint.generation_layer.generator.coref_generator import CorefGenerator
from textflint.input_layer.dataset import Dataset
import unittest
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


class TestRndDelete(unittest.TestCase):

    def test_transform(self):
        gene = CorefGenerator(trans_methods=["RndDelete"],
                              sub_methods=[])
        for original_samples, trans_rst, trans_type in gene.generate(dataset):
            self.assertEqual(len(original_samples), len(trans_rst))
            for so, st in zip(original_samples, trans_rst):
                if so.num_sentences() >= 2:
                    self.assertTrue(so.num_sentences()-1 >= st.num_sentences())  


if __name__ == "__main__":
    unittest.main()

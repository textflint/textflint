import unittest
from textflint.input_layer.dataset import Dataset
from textflint.generation_layer.generator.sa_generator import SAGenerator

sample1 = {'x': 'Titanic is my favorite movie.',
           'y': 'pos'}
sample2 = {'x': 'I don\'t like the actor Tim Hill', 'y': 'neg'}
sample3 = {'x': 'The leading actor is good.',
           'y': 'pos'}
sample4 = {'x': '',
           'y': 'pos'}
sample5 = {'x': '!@#$$%^&*()_+}{|":?><',
           'y': 'pos'}
single_data_sample = [sample1]
data_samples = [sample1, sample2, sample3, sample4, sample5]
dataset = Dataset('SA')
single_dataset = Dataset('SA')
dataset.load(data_samples)
single_dataset.load(single_data_sample)


class TestSpecialEntityTyposSwap(unittest.TestCase):

    def test_generate(self):
        # test task transformation
        trans_methods = ["SwapSpecialEnt", "AddSum",
                             "DoubleDenial", "SwapNum"]
        SA_config = {'AddSum': [{'entity_type': 'movie'},
                                {'entity_type': 'person'}],
                     'SwapSpecialEnt': [{'entity_type': 'movie'},
                                        {'entity_type': 'person'}]}
        gene = SAGenerator(trans_methods=trans_methods,
                           sub_methods=[],
                           trans_config=SA_config)

        for original_samples, trans_rst, trans_type in gene.generate(dataset):
            for index in range(len(original_samples)):
                self.assertEqual(original_samples[index].y, trans_rst[index].y)

        # test wrong trans_methods
        gene = SAGenerator(trans_methods=["wrong_transform_method"],
                           sub_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = SAGenerator(trans_methods=["AddSubtree"],
                           sub_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = SAGenerator(trans_methods=["OOV"],
                           sub_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = SAGenerator(trans_methods="ReverseNeg",
                           sub_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))

        # test pipeline trans_methods
        trans_methods = [["SwapSpecialEnt", "AddSum"],
                             ["SwapSpecialEnt", "DoubleDenial"]]
        gene = SAGenerator(trans_methods=trans_methods,
                           sub_methods=[],
                           trans_config=SA_config)
        for original_samples, trans_rst, trans_type in gene.generate(dataset):
            self.assertEqual(len(trans_rst), len(original_samples))


if __name__ == "__main__":
    unittest.main()
